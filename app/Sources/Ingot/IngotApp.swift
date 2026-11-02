import AppKit
import ServiceManagement
import SwiftUI

@main
struct IngotApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @StateObject private var engine = Engine.shared

    var body: some Scene {
        Window("Ingot", id: "main") {
            ContentView(engine: engine)
                .frame(minWidth: 760, minHeight: 520)
                .ignoresSafeArea()
                .background(Color(nsColor: .windowBackgroundColor))
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1180, height: 780)

        // SwiftUI writes this binding on every update, so an unchanged value
        // must not publish, or it would update forever.
        let inserted = Binding(
            get: { engine.showsMenuBarItem },
            set: { if $0 != engine.showsMenuBarItem { engine.showsMenuBarItem = $0 } })
        MenuBarExtra("Ingot", systemImage: "cube.fill", isInserted: inserted) {
            StatusPanel(engine: engine)
        }
        .menuBarExtraStyle(.window)
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        Engine.shared.start()
    }
    // With the menu bar item, closing the window keeps the server running for
    // connected apps.
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        !Engine.shared.showsMenuBarItem
    }
    func applicationWillTerminate(_ notification: Notification) {
        Engine.shared.stop()
    }
}

struct ContentView: View {
    @ObservedObject var engine: Engine

    var body: some View {
        ZStack(alignment: .top) {
            switch engine.state {
            case .starting:
                VStack(spacing: 12) {
                    ProgressView()
                    Text("Starting Ingot…").foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            case .ready(let url):
                WebView(url: url, engine: engine)
            case .failed(let message):
                VStack(spacing: 12) {
                    Image(systemName: "exclamationmark.triangle").font(.largeTitle)
                    Text("Ingot could not start").font(.title2)
                    Text(message)
                        .font(.system(.callout, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: 560)
                    HStack {
                        Button("Try Again") { engine.start() }.keyboardShortcut(.defaultAction)
                        Button("Show Log") { engine.showLog() }
                    }
                }
                .padding(32)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            // The page covers the title bar, and web content cannot move a window.
            Color.clear
                .frame(height: 30)
                .contentShape(Rectangle())
                .gesture(WindowDragGesture())
        }
    }
}

/// Panel under the menu bar icon.
struct StatusPanel: View {
    @ObservedObject var engine: Engine
    @Environment(\.openWindow) private var openWindow
    @State private var startsAtLogin = SMAppService.mainApp.status == .enabled
    private let gold = Color(red: 0.96, green: 0.76, blue: 0.35)

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Ingot").font(.headline)
                Label(engine.ready ? "Ready for your apps" : "Not running", systemImage: "circle.fill")
                    .labelStyle(StatusLabel(color: engine.ready ? .green : .secondary))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            VStack(alignment: .leading, spacing: 6) {
                Text("Loaded model").font(.caption).foregroundStyle(.secondary)
                Text(engine.modelTitle ?? "None").font(.body.weight(.semibold))
                if engine.memoryTotal > 0 {
                    ProgressView(value: min(engine.memoryUsed, engine.memoryTotal), total: engine.memoryTotal)
                        .tint(gold)
                    Text(String(format: "%.0f of %.0f GiB in use", engine.memoryUsed, engine.memoryTotal))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 12))
            if let address = engine.address {
                HStack {
                    Text(address).font(.system(.caption, design: .monospaced)).lineLimit(1)
                    Spacer()
                    Button("Copy") { engine.copyAddress() }
                }
            }
            Button {
                openWindow(id: "main")
                NSApp.activate()
            } label: {
                Text("Open Ingot").font(.body.weight(.semibold)).frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .tint(gold)
            .foregroundStyle(Color(red: 0.14, green: 0.1, blue: 0.02))
            .controlSize(.large)
            Divider()
            Toggle("Start at Login", isOn: $startsAtLogin)
                .onChange(of: startsAtLogin) { _, on in
                    do {
                        if on {
                            try SMAppService.mainApp.register()
                        } else {
                            try SMAppService.mainApp.unregister()
                        }
                    } catch {
                        startsAtLogin = SMAppService.mainApp.status == .enabled
                        Engine.alert("Start at Login is not available", error.localizedDescription)
                    }
                }
            Button("Install Command Line Tool") { engine.installCommandLineTool() }
                .buttonStyle(.plain)
            Button("Show Log") { engine.showLog() }
                .buttonStyle(.plain)
            Button("Quit Ingot") { NSApp.terminate(nil) }
                .buttonStyle(.plain)
        }
        .padding(16)
        .frame(width: 300)
    }
}

private struct StatusLabel: LabelStyle {
    let color: Color
    func makeBody(configuration: Configuration) -> some View {
        HStack(spacing: 5) {
            configuration.icon.font(.system(size: 7)).foregroundStyle(color)
            configuration.title
        }
    }
}
