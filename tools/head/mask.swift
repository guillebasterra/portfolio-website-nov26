// Subject mask for each photo using Apple's Vision framework.
// usage: swift mask.swift <out_dir> <image>...
import Foundation
import Vision
import CoreImage
import AppKit

let args = CommandLine.arguments
let outDir = URL(fileURLWithPath: args[1])
let ctx = CIContext()
for path in args.dropFirst(2) {
    let url = URL(fileURLWithPath: path)
    guard let src = CIImage(contentsOf: url, options: [.applyOrientationProperty: true]) else { print("skip \(path)"); continue }
    let handler = VNImageRequestHandler(ciImage: src)
    let req = VNGenerateForegroundInstanceMaskRequest()
    try handler.perform([req])
    guard let obs = req.results?.first else { print("no subject \(path)"); continue }
    let buf = try obs.generateScaledMaskForImage(forInstances: obs.allInstances, from: handler)
    let mask = CIImage(cvPixelBuffer: buf)
    let out = outDir.appendingPathComponent(url.deletingPathExtension().lastPathComponent + ".png")
    try ctx.writePNGRepresentation(of: mask, to: out, format: .L8, colorSpace: CGColorSpace(name: CGColorSpace.linearGray)!)
    print("\(out.lastPathComponent) \(Int(mask.extent.width))x\(Int(mask.extent.height))")
}
