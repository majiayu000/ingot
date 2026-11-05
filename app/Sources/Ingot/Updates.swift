import Sparkle

/// Updates come from the appcast attached to the latest GitHub release. Sparkle
/// checks the EdDSA signature and the Developer ID before it installs one.
@MainActor
enum Updates {
    // Sparkle needs a real app bundle, so `swift run` builds go without it.
    static let controller: SPUStandardUpdaterController? =
        Bundle.main.bundleURL.pathExtension == "app"
        ? SPUStandardUpdaterController(startingUpdater: true, updaterDelegate: nil, userDriverDelegate: nil)
        : nil

    static func check() { controller?.checkForUpdates(nil) }

    static func setAutomatic(_ on: Bool) {
        guard let updater = controller?.updater, updater.automaticallyChecksForUpdates != on else { return }
        updater.automaticallyChecksForUpdates = on
    }
}
