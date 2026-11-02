// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "Ingot",
    platforms: [.macOS(.v15)],
    targets: [
        .executableTarget(name: "Ingot", path: "Sources/Ingot", swiftSettings: [.swiftLanguageMode(.v5)])
    ]
)
