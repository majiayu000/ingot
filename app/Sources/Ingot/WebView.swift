import AppKit
import ServiceManagement
import SwiftUI
import WebKit

/// The interface is the page the engine serves, so the app and a browser
/// always show the same thing.
struct WebView: NSViewRepresentable {
    let url: URL
    let engine: Engine

    func makeCoordinator() -> Coordinator { Coordinator(url: url, engine: engine) }

    func makeNSView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.userContentController.add(context.coordinator, name: "ingot")
        if !engine.apiKey.isEmpty,
            let key = try? JSONSerialization.data(withJSONObject: engine.apiKey, options: .fragmentsAllowed),
            let text = String(data: key, encoding: .utf8)
        {
            let script = WKUserScript(
                source: "window.ingotKey = \(text);", injectionTime: .atDocumentStart, forMainFrameOnly: true)
            configuration.userContentController.addUserScript(script)
        }
        let login = SMAppService.mainApp.status == .enabled
        configuration.userContentController.addUserScript(
            WKUserScript(
                source: "window.ingotLogin = \(login);", injectionTime: .atDocumentStart, forMainFrameOnly: true))
        let view = WKWebView(frame: .zero, configuration: configuration)
        view.underPageBackgroundColor = .windowBackgroundColor
        context.coordinator.view = view
        view.navigationDelegate = context.coordinator
        view.uiDelegate = context.coordinator
        view.load(URLRequest(url: url))
        return view
    }

    func updateNSView(_ view: WKWebView, context: Context) {
        if context.coordinator.url != url {
            context.coordinator.url = url
            view.load(URLRequest(url: url))
        }
    }

    final class Coordinator: NSObject, WKNavigationDelegate, WKUIDelegate, WKScriptMessageHandler {
        var url: URL
        let engine: Engine
        weak var view: WKWebView?

        init(url: URL, engine: Engine) {
            self.url = url
            self.engine = engine
        }

        // What the page cannot do itself.
        func userContentController(
            _ controller: WKUserContentController, didReceive message: WKScriptMessage
        ) {
            guard message.frameInfo.isMainFrame, let body = message.body as? [String: Any],
                let action = body["action"] as? String
            else { return }
            switch action {
            case "settings":
                engine.reloadSettings()
            case "restart":
                engine.restart()
            case "export":
                if let text = body["text"] as? String {
                    engine.exportChats(text, name: body["name"] as? String ?? "ingot-chats.json")
                }
            case "terminal":
                if let tool = body["tool"] as? String, let model = body["model"] as? String {
                    engine.launchInTerminal(tool: tool, model: model)
                }
            case "chooseFolder":
                chooseFolder()
            case "checkUpdates":
                Updates.check()
            case "login":
                if let on = body["on"] as? Bool { openAtLogin(on) }
            default:
                break
            }
        }

        private func openAtLogin(_ on: Bool) {
            var failure = "null"
            do {
                if on {
                    try SMAppService.mainApp.register()
                } else {
                    try SMAppService.mainApp.unregister()
                }
            } catch {
                if let data = try? JSONSerialization.data(
                    withJSONObject: error.localizedDescription, options: .fragmentsAllowed),
                    let text = String(data: data, encoding: .utf8)
                {
                    failure = text
                }
            }
            let enabled = SMAppService.mainApp.status == .enabled
            view?.evaluateJavaScript("window.ingotLoginChanged(\(enabled), \(failure))")
        }

        private func chooseFolder() {
            let panel = NSOpenPanel()
            panel.canChooseFiles = false
            panel.canChooseDirectories = true
            panel.canCreateDirectories = true
            panel.begin { [weak self] response in
                guard response == .OK, let path = panel.url?.path,
                    let data = try? JSONSerialization.data(withJSONObject: path, options: .fragmentsAllowed),
                    let text = String(data: data, encoding: .utf8)
                else { return }
                self?.view?.evaluateJavaScript("window.ingotFolderChosen(\(text))")
            }
        }

        // Links in answers open in the browser, not in place of the chat.
        func webView(
            _ webView: WKWebView, decidePolicyFor action: WKNavigationAction,
            decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
        ) {
            if let target = action.request.url, target.host != url.host || target.port != url.port {
                NSWorkspace.shared.open(target)
                decisionHandler(.cancel)
            } else {
                decisionHandler(.allow)
            }
        }

        func webView(
            _ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration,
            for action: WKNavigationAction, windowFeatures: WKWindowFeatures
        ) -> WKWebView? {
            if let target = action.request.url { NSWorkspace.shared.open(target) }
            return nil
        }

        // Without this the page's Image button does nothing.
        func webView(
            _ webView: WKWebView, runOpenPanelWith parameters: WKOpenPanelParameters,
            initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping ([URL]?) -> Void
        ) {
            let panel = NSOpenPanel()
            panel.allowsMultipleSelection = parameters.allowsMultipleSelection
            panel.canChooseDirectories = false
            panel.begin { completionHandler($0 == .OK ? panel.urls : nil) }
        }
    }
}
