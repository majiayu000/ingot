import AppKit
import CFNetwork
import Foundation

/// Runs the bundled `ingot serve` and reports its state to the interface.
@MainActor
final class Engine: ObservableObject {
    static let shared = Engine()

    enum State {
        case starting
        case ready(URL)
        case failed(String)
    }

    /// First port tried. Ollama's own port stays free for Ollama.
    private static let firstPort: UInt16 = 11435
    private static let pidKey = "enginePID"
    /// Restarts after a crash before the error screen is shown.
    private static let maxRestarts = 3

    @Published private(set) var state = State.starting
    @Published private(set) var ready = false
    @Published private(set) var address: String?
    @Published private(set) var modelTitle: String?
    @Published private(set) var memoryUsed = 0.0
    @Published private(set) var memoryTotal = 0.0
    @Published var showsMenuBarItem = true

    private var process: Process?
    private var statusTask: Task<Void, Never>?
    private var restarts = 0
    private var stopping = false
    private var restarting = false
    /// API key the page and status checks send.
    private(set) var apiKey = ""

    private let binary = Bundle.main.resourceURL!.appendingPathComponent("ingot")
    private let log = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Logs/Ingot/ingot.log")
    private let settingsFile: URL = {
        if let directory = ProcessInfo.processInfo.environment["INGOT_SUPPORT"] {
            return URL(fileURLWithPath: directory).appendingPathComponent("settings.json")
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/Ingot/settings.json")
    }()

    func start() {
        guard process == nil else { return }
        state = .starting
        ready = false
        stopping = false
        stopLeftover()
        let settings = readSettings()
        apply(settings)
        let preferred = (settings["port"] as? Int).flatMap(UInt16.init(exactly:)) ?? Self.firstPort
        guard let port = Self.freePort(from: preferred) else {
            return fail("No free local port near \(preferred).")
        }
        do {
            let task = try launch(port: port, settings: settings)
            process = task
            UserDefaults.standard.set(Int(task.processIdentifier), forKey: Self.pidKey)
        } catch {
            return fail("Cannot start the engine: \(error.localizedDescription)")
        }
        let root = URL(string: "http://127.0.0.1:\(port)/")!
        Task { await self.waitUntilReady(root) }
    }

    /// Starts the engine again with the saved settings.
    func restart() {
        guard let process else { return start() }
        restarting = true
        statusTask?.cancel()
        state = .starting
        process.terminate()
    }

    /// Applies the settings the app itself uses.
    func reloadSettings() {
        apply(readSettings())
    }

    func stop() {
        stopping = true
        statusTask?.cancel()
        process?.terminate()
        process?.waitUntilExit()
        UserDefaults.standard.removeObject(forKey: Self.pidKey)
    }

    func copyAddress() {
        guard let address else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(address, forType: .string)
    }

    func showLog() {
        NSWorkspace.shared.activateFileViewerSelecting([log])
    }

    /// Opens Terminal running `ingot launch`. Only known tools get here, and
    /// the model name is quoted for the shell.
    func launchInTerminal(tool: String, model: String) {
        guard ["claude", "codex", "opencode"].contains(tool) else { return }
        let quoted = "'" + model.replacingOccurrences(of: "'", with: "'\\''") + "'"
        let script = "#!/bin/zsh -l\n'\(binary.path)' launch \(tool) --model \(quoted)\n"
        let file = FileManager.default.temporaryDirectory
            .appendingPathComponent("ingot-launch-\(tool).command")
        do {
            try script.write(to: file, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: file.path)
            NSWorkspace.shared.open(file)
        } catch {
            Self.alert("Could not open Terminal", error.localizedDescription)
        }
    }

    func exportChats(_ text: String) {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "ingot-chats.json"
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            do {
                try text.write(to: url, atomically: true, encoding: .utf8)
            } catch {
                Self.alert("Could not save the file", error.localizedDescription)
            }
        }
    }

    func installCommandLineTool() {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let directory = home.appendingPathComponent(".local/bin")
        let link = directory.appendingPathComponent("ingot")
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            if let target = try? FileManager.default.destinationOfSymbolicLink(atPath: link.path) {
                if target != binary.path { try FileManager.default.removeItem(at: link) }
            } else if FileManager.default.fileExists(atPath: link.path) {
                return Self.alert("Not installed", "\(link.path) already exists and was left alone.")
            }
            if !FileManager.default.fileExists(atPath: link.path) {
                try FileManager.default.createSymbolicLink(at: link, withDestinationURL: binary)
            }
            Self.alert("Command line tool installed", "\(link.path) now runs Ingot. If `ingot` is not found, add ~/.local/bin to your PATH.")
        } catch {
            Self.alert("Not installed", error.localizedDescription)
        }
    }

    static func alert(_ title: String, _ text: String) {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = text
        alert.runModal()
    }

    private func fail(_ message: String) {
        process = nil
        address = nil
        ready = false
        state = .failed(message)
    }

    private func readSettings() -> [String: Any] {
        guard let data = try? Data(contentsOf: settingsFile),
            let value = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return [:] }
        return value
    }

    private func apply(_ settings: [String: Any]) {
        let inserted = settings["menu_bar"] as? Bool ?? true
        if inserted != showsMenuBarItem { showsMenuBarItem = inserted }
        apiKey = settings["api_key"] as? String ?? ""
    }

    private func arguments(port: UInt16, settings: [String: Any]) -> [String] {
        var arguments = ["serve", "--port", String(port)]
        let keepAlive = settings["keep_alive"] as? String ?? "10m"
        if keepAlive != "-1" { arguments += ["--keep-alive", keepAlive] }
        if let context = settings["context"] as? Int, context > 0 {
            arguments += ["--context", String(context)]
        }
        if let model = settings["default_model"] as? String, !model.isEmpty {
            arguments += ["--default", model]
        }
        if settings["allow_network"] as? Bool == true { arguments += ["--host", "0.0.0.0"] }
        return arguments
    }

    private func launch(port: UInt16, settings: [String: Any]) throws -> Process {
        try FileManager.default.createDirectory(
            at: log.deletingLastPathComponent(), withIntermediateDirectories: true)
        if !FileManager.default.fileExists(atPath: log.path) {
            FileManager.default.createFile(atPath: log.path, contents: nil)
        }
        let output = try FileHandle(forWritingTo: log)
        try output.seekToEnd()
        let task = Process()
        task.executableURL = binary
        task.arguments = arguments(port: port, settings: settings)
        var environment = Self.environment()
        if !apiKey.isEmpty { environment["INGOT_API_KEY"] = apiKey }
        task.environment = environment
        task.standardOutput = output
        task.standardError = output
        task.terminationHandler = { [weak self] finished in
            Task { @MainActor in self?.terminated(finished) }
        }
        try task.run()
        return task
    }

    private func terminated(_ finished: Process) {
        guard finished === process, !stopping else { return }
        process = nil
        statusTask?.cancel()
        if restarting {
            restarting = false
            restarts = 0
            return start()
        }
        let reason = "The engine stopped (exit code \(finished.terminationStatus)).\n\(lastLogLines())"
        guard restarts < Self.maxRestarts else { return fail(reason) }
        restarts += 1
        start()
    }

    private func waitUntilReady(_ root: URL) async {
        for _ in 0..<150 {
            guard process != nil else { return }
            if let (_, response) = try? await URLSession.shared.data(from: root),
                (response as? HTTPURLResponse)?.statusCode == 200
            {
                address = root.appendingPathComponent("v1").absoluteString
                state = .ready(root)
                ready = true
                restarts = 0
                watchStatus(root)
                return
            }
            try? await Task.sleep(for: .milliseconds(200))
        }
        process?.terminate()
        fail("The engine did not answer in 30 seconds.\n\(lastLogLines())")
    }

    private func watchStatus(_ root: URL) {
        statusTask?.cancel()
        var request = URLRequest(url: root.appendingPathComponent("v1/status"))
        if !apiKey.isEmpty { request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization") }
        statusTask = Task {
            while !Task.isCancelled {
                if let (data, _) = try? await URLSession.shared.data(for: request),
                    let status = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
                {
                    let memory = status["memory"] as? [String: Any]
                    let bytes = (memory?["metal_allocated_bytes"] as? Double) ?? 0
                    let total = (memory?["recommended_working_set_bytes"] as? Double) ?? 0
                    modelTitle = status["model"] as? String
                    memoryUsed = bytes / 1_073_741_824
                    memoryTotal = total / 1_073_741_824
                }
                try? await Task.sleep(for: .seconds(3))
            }
        }
    }

    private func lastLogLines() -> String {
        guard let text = try? String(contentsOf: log, encoding: .utf8) else { return "" }
        return text.split(separator: "\n").suffix(6).joined(separator: "\n")
    }

    /// An engine left by a crash of this app would keep its port and memory.
    private func stopLeftover() {
        let pid = pid_t(UserDefaults.standard.integer(forKey: Self.pidKey))
        guard pid > 0 else { return }
        var path = [CChar](repeating: 0, count: Int(MAXPATHLEN))
        if proc_pidpath(pid, &path, UInt32(path.count)) > 0, String(cString: path) == binary.path {
            kill(pid, SIGTERM)
            usleep(500_000)
        }
        UserDefaults.standard.removeObject(forKey: Self.pidKey)
    }

    private static func freePort(from first: UInt16) -> UInt16? {
        (first..<(first &+ 20)).first(where: isFree)
    }

    private static func isFree(_ port: UInt16) -> Bool {
        let socket = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        guard socket >= 0 else { return false }
        defer { close(socket) }
        // A port this app just left is free, whatever the kernel's linger state.
        var on: Int32 = 1
        setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &on, socklen_t(MemoryLayout<Int32>.size))
        var address = sockaddr_in()
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = port.bigEndian
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        return withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(socket, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) == 0
            }
        }
    }

    /// Apps opened from the Finder have no shell settings, so model downloads
    /// would ignore a proxy set in System Settings without this.
    private static func environment() -> [String: String] {
        var environment = ProcessInfo.processInfo.environment
        guard environment["HTTPS_PROXY"] == nil, environment["https_proxy"] == nil,
            let settings = CFNetworkCopySystemProxySettings()?.takeRetainedValue() as? [String: Any],
            settings[kCFNetworkProxiesHTTPSEnable as String] as? Int == 1,
            let host = settings[kCFNetworkProxiesHTTPSProxy as String] as? String,
            let port = settings[kCFNetworkProxiesHTTPSPort as String] as? Int
        else { return environment }
        environment["HTTPS_PROXY"] = "http://\(host):\(port)"
        return environment
    }
}
