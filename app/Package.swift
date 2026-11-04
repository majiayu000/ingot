// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "Ingot",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/sparkle-project/Sparkle", exact: "2.10.0")
    ],
    targets: [
        .executableTarget(
            name: "Ingot", dependencies: [.product(name: "Sparkle", package: "Sparkle")], path: "Sources/Ingot",
            swiftSettings: [.swiftLanguageMode(.v5)],
            linkerSettings: [.unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", "@executable_path/../Frameworks"])])
    ]
)
