// Draws the app icon and the disk image background.
// usage: swift make-art.swift icon OUT.png | background OUT.png
import AppKit

func render(_ width: Int, _ height: Int, to path: String, draw: (CGContext) -> Void) {
    let context = CGContext(
        data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpace(name: CGColorSpace.sRGB)!,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    draw(context)
    let data = NSBitmapImageRep(cgImage: context.makeImage()!)
        .representation(using: .png, properties: [:])!
    try! data.write(to: URL(fileURLWithPath: path))
}

func color(_ r: Double, _ g: Double, _ b: Double) -> CGColor {
    CGColor(srgbRed: r, green: g, blue: b, alpha: 1)
}

func gradient(_ c: CGContext, _ path: CGPath, _ top: CGColor, _ bottom: CGColor, from: CGPoint, to: CGPoint) {
    c.saveGState()
    c.addPath(path)
    c.clip()
    let g = CGGradient(colorsSpace: nil, colors: [top, bottom] as CFArray, locations: [0, 1])!
    c.drawLinearGradient(g, start: from, end: to, options: [.drawsBeforeStartLocation, .drawsAfterEndLocation])
    c.restoreGState()
}

func polygon(_ points: [(Double, Double)]) -> CGPath {
    let p = CGMutablePath()
    p.addLines(between: points.map { CGPoint(x: $0.0, y: $0.1) })
    p.closeSubpath()
    return p
}

func icon(_ c: CGContext) {
    // Standard macOS icon grid: 824 of 1024, with a soft shadow.
    let tile = CGPath(
        roundedRect: CGRect(x: 100, y: 100, width: 824, height: 824), cornerWidth: 186, cornerHeight: 186,
        transform: nil)
    c.saveGState()
    c.setShadow(offset: CGSize(width: 0, height: -14), blur: 34, color: CGColor(gray: 0, alpha: 0.4))
    c.addPath(tile)
    c.setFillColor(color(0.11, 0.13, 0.19))
    c.fillPath()
    c.restoreGState()
    gradient(c, tile, color(0.20, 0.23, 0.33), color(0.07, 0.08, 0.13), from: CGPoint(x: 512, y: 924), to: CGPoint(x: 512, y: 100))

    // A gold bar with a lit top, a front face and a darker side.
    let top = polygon([(274, 612), (672, 612), (780, 698), (382, 698)])
    let front = polygon([(274, 612), (672, 612), (604, 330), (206, 330)])
    let side = polygon([(672, 612), (780, 698), (820, 416), (604, 330)])
    c.saveGState()
    c.setShadow(offset: CGSize(width: 0, height: -26), blur: 60, color: CGColor(gray: 0, alpha: 0.55))
    c.addPath(front)
    c.addPath(side)
    c.setFillColor(color(0.6, 0.4, 0.1))
    c.fillPath()
    c.restoreGState()
    gradient(c, front, color(1.0, 0.82, 0.38), color(0.84, 0.54, 0.13), from: CGPoint(x: 300, y: 612), to: CGPoint(x: 420, y: 330))
    gradient(c, side, color(0.78, 0.5, 0.13), color(0.5, 0.29, 0.06), from: CGPoint(x: 680, y: 600), to: CGPoint(x: 800, y: 400))
    gradient(c, top, color(1.0, 0.95, 0.72), color(1.0, 0.84, 0.42), from: CGPoint(x: 560, y: 698), to: CGPoint(x: 500, y: 612))
    // Highlight along the bar's front edge.
    c.setStrokeColor(CGColor(srgbRed: 1, green: 0.97, blue: 0.82, alpha: 0.9))
    c.setLineWidth(5)
    c.setLineCap(.round)
    c.move(to: CGPoint(x: 280, y: 612))
    c.addLine(to: CGPoint(x: 668, y: 612))
    c.strokePath()
}

func background(_ c: CGContext) {
    c.setFillColor(color(0.965, 0.965, 0.975))
    c.fill(CGRect(x: 0, y: 0, width: 1320, height: 800))
    // Arrow from the app to the Applications folder (2x pixels).
    c.setStrokeColor(color(0.62, 0.64, 0.7))
    c.setLineWidth(12)
    c.setLineCap(.round)
    c.setLineJoin(.round)
    c.move(to: CGPoint(x: 560, y: 410))
    c.addLine(to: CGPoint(x: 760, y: 410))
    c.move(to: CGPoint(x: 710, y: 460))
    c.addLine(to: CGPoint(x: 760, y: 410))
    c.addLine(to: CGPoint(x: 710, y: 360))
    c.strokePath()
}

let arguments = CommandLine.arguments
switch arguments.dropFirst().first {
case "icon": render(1024, 1024, to: arguments[2], draw: icon)
case "background": render(1320, 800, to: arguments[2], draw: background)
default: fatalError("usage: make-art.swift icon|background OUT.png")
}
