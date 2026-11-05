# Draft formula. Replace the placeholder before publishing:
#   SHA256          contents of ingot-<version>-aarch64-apple-darwin.tar.gz.sha256
class Ingot < Formula
  desc "Run AI models locally on Apple Silicon Macs"
  homepage "https://github.com/majiayu000/ingot"
  version "0.1.0"
  url "https://github.com/majiayu000/ingot/releases/download/v#{version}/ingot-#{version}-aarch64-apple-darwin.tar.gz"
  sha256 "SHA256"
  license "MIT"

  depends_on arch: :arm64
  depends_on macos: :sequoia

  def install
    bin.install "ingot"
    doc.install "README.md", "CHANGELOG.md", "NOTICE"
  end

  test do
    assert_match "run AI models locally", shell_output("#{bin}/ingot --help")
  end
end
