// swift-tools-version: 5.7.3
import PackageDescription
let package = Package(
    name: "alta",
    dependencies:[
        .package(url: "https://github.com/pvieito/PythonKit.git", .branch("master"))
    ],
    targets : [
        .target(
            name:"alta",
            dependencies:["PythonKit"]
        ),
    ]
)