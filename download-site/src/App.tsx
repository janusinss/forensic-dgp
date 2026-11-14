import releaseData from './release.generated.json'

type Release = {
  version: string
  bytes: number
  sha256: string
  signed: boolean
  preview: boolean
  downloadUrl: string
  checksumUrl?: string
  filename: string
  offline?: boolean
}

const release = releaseData as Release
const size = release.bytes >= 1_000_000_000 ? `${(release.bytes / 1_000_000_000).toFixed(2)} GB` : `${(release.bytes / 1_000_000).toFixed(1)} MB`
const [major, minor, patch] = release.version.split('.').map(Number)
const newerMinor = major > 0 || minor > 2
const currentWorkflow = newerMinor || (minor === 2 && patch >= 6)
const updateCheck = newerMinor || (minor === 2 && patch >= 7)
const solverSuite = newerMinor || (minor === 2 && patch >= 8)
const manualCrop = newerMinor || (minor === 2 && patch >= 9)
const alignedReference = newerMinor || (minor === 2 && patch >= 10)

function DownloadIcon() {
  return (
    <svg aria-hidden="true" width="20" height="20" viewBox="0 0 20 20" fill="none">
      <path d="M10 2.5v10m-4-4 4 4 4-4M3.5 13v3.5h13V13" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

export default function App() {
  return (
    <>
      <a className="skip-link" href="#main">Skip to content</a>
      <div className="page-shell">
        <header className="site-header">
          <a className="wordmark" href="#main" aria-label="Forensic DGP home">
            <svg className="wordmark-symbol" aria-hidden="true" width="26" height="26" viewBox="0 0 26 26" fill="none">
              <path d="M3 9V3h6m8 0h6v6M3 17v6h6m8 0h6v-6" stroke="currentColor" strokeWidth="2" />
              <circle cx="13" cy="13" r="3" fill="currentColor" />
            </svg>
            <span>Forensic <strong>DGP</strong></span>
          </a>
          <a className="header-link" href="#setup">Get started <span aria-hidden="true">↗</span></a>
        </header>

        <main id="main">
          <section className="hero" aria-labelledby="hero-title">
            <div className="hero-copy">
              <h1 id="hero-title">A closer look.<br />On your own PC.</h1>
              <p className="hero-description">{currentWorkflow
                ? 'Restore a face from a picture or select several faces from a frozen video frame. Everything runs on your PC.'
                : 'Capture a face from video, review your picture, and explore restoration locally.'}</p>
              <div className="download-group">
                <a className="download-button" href={release.downloadUrl} download={release.filename}>
                  <DownloadIcon />
                  <span>Download for Windows</span>
                  <span className="download-arrow" aria-hidden="true">↗</span>
                </a>
                <p className="download-details">Windows 10 / 11 <span aria-hidden="true">·</span> 64-bit <span aria-hidden="true">·</span> {size}</p>
                <p className="release-details">Version {release.version}{release.preview ? ' preview' : ''}{release.offline ? ' · Offline installer' : ''}{!release.signed ? ' · Unsigned installer' : ''}</p>
              </div>
            </div>

            <div className="brand-visual" aria-hidden="true">
              <span className="frame-corner corner-top-left" />
              <span className="frame-corner corner-top-right" />
              <div className="speech-block speech-picture">Your<br />picture.</div>
              <div className="speech-block speech-closer">Let’s look<br />closer.</div>
              <p className="visual-signature">A closer look starts here.</p>
              <span className="frame-corner corner-bottom-left" />
              <span className="frame-corner corner-bottom-right" />
            </div>
          </section>

          <section className="setup-section" id="setup" aria-labelledby="setup-title">
            <div className="setup-heading">
              <h2 id="setup-title">From download to desktop.</h2>
              <p>Start with your own picture or video.</p>
            </div>
            <ol className="setup-steps">
              <li>
                <span className="step-number" aria-hidden="true">01</span>
                <div>
                  <h3>Download</h3>
                  <p>Save the Windows installer to your PC.</p>
                </div>
              </li>
              <li>
                <span className="step-number" aria-hidden="true">02</span>
                <div>
                  <h3>Install</h3>
                  <p>Choose a folder for the app, models and saved records.</p>
                </div>
              </li>
              <li>
                <span className="step-number" aria-hidden="true">03</span>
                <div>
                  <h3>Open</h3>
                  <p>Launch Forensic DGP from your Start Menu.</p>
                </div>
              </li>
            </ol>
            <p className="setup-note">{release.offline
              ? 'Includes Python, application libraries and models. After downloading, setup works offline. Microsoft Edge is required; setup checks available disk space. Existing installations can be updated, repaired or moved.'
              : 'First setup needs internet access, Microsoft Edge and at least 4 GB of free space. Setup shows download progress for Python and approximately 720 MiB of models.'}</p>
            {updateCheck && (
              <p className="setup-note">{alignedReference
                ? 'Use Check for updates on the installer’s Before you install page. To upgrade, run the newer installer in your existing folder to keep your saved records. This release includes Windows PE version, DPI-aware and long-path launcher metadata, quiet helper execution, and fixes startup when Windows uses different spellings of the same temporary folder.'
                : 'Use Check for updates on the installer’s Before you install page. To upgrade, run the newer installer in your existing folder to keep your saved records. This release also fixes startup when Windows uses different spellings of the same temporary folder.'}</p>
            )}
            {(release.version === '0.2.4' || currentWorkflow) && (
              <p className="setup-note">Uninstalling permanently removes the app, models and saved records from the installation folder.</p>
            )}
          </section>

          {currentWorkflow && (
            <section className="setup-section" aria-labelledby="workflow-title">
              <div className="setup-heading">
                <h2 id="workflow-title">Restore and compare</h2>
                <p>Workflow in version {release.version}</p>
              </div>
              <ol className="setup-steps">
                <li>
                  <span className="step-number" aria-hidden="true">01</span>
                  <div>
                    <h3>Choose the faces</h3>
                    <p>{manualCrop
                      ? 'Upload a picture or freeze a video frame. Picture selects Face 1 automatically or lets you draw a Manual crop; on video, select one detected face or use Add all to select up to 12 faces from the frame.'
                      : solverSuite
                        ? 'Upload a picture or freeze a video frame. Picture selects Face 1 automatically; on video, select one detected face or use Add all to select up to 12 faces from the frame.'
                        : 'Upload a picture or freeze a video frame. Select one detected face or use Add all to select up to 12 faces from the frame.'}</p>
                  </div>
                </li>
                <li>
                  <span className="step-number" aria-hidden="true">02</span>
                  <div>
                    <h3>Restore with Auto</h3>
                    <p>{solverSuite
                      ? 'Auto handles regular, tilted, perspective and irregular mosaics, scanlines, 8 × 8 / 16 × 16 macroblocks, localized blur, noise, color cast or haze, ringing, banding, aliasing and color bleeding before restoring each face.'
                      : updateCheck
                        ? 'Auto handles regular, tilted and irregular mosaics, scanlines, JPEG blocking and localized blur. When it finds separate evidence for overlapping artifacts, it can combine treatments before restoring each face.'
                        : 'Auto checks for pixelation, scanlines and localized blur. It estimates the smoothing strength where applicable. The app restores the selected crops one at a time.'}</p>
                  </div>
                </li>
                <li>
                  <span className="step-number" aria-hidden="true">03</span>
                  <div>
                    <h3>Compare and download</h3>
                    <p>{alignedReference
                      ? 'Switch between faces to compare Before, Prepared and After, native noise and edge diagnostics, and PNG downloads. Add an aligned clean reference or padded crop to evaluate exact same-frame Before, Prepared and After MAE and PSNR.'
                      : manualCrop
                        ? 'Switch between faces to compare Before, Prepared and After, native noise and edge diagnostics, and PNG downloads. Add an aligned clean reference or padded crop to see MAE and PSNR.'
                        : 'Switch between faces to see Before and After, image diagnostics and PNG downloads. Add an aligned clean reference to see MAE and PSNR.'}</p>
                  </div>
                </li>
              </ol>
              <p className="setup-note">Manual lets you mark areas to protect or blur in pictures. Off keeps the resized original.</p>
            </section>
          )}

          <aside className="research-note" aria-label="Experimental use">
            <p>Made for a closer look. Restored details are experimental estimates and cannot verify someone’s identity.</p>
            <a href="/LICENSES.txt">Non-commercial use · CodeFormer terms <span aria-hidden="true">↗</span></a>
          </aside>
        </main>

        <footer className="site-footer">
          <p>Forensic DGP</p>
          <div className="footer-links">
            <a href={release.checksumUrl ?? `${release.downloadUrl}.sha256`} download={`${release.filename}.sha256`}>Download checksum <span aria-hidden="true">↗</span></a>
            <a href="/LICENSES.txt">Licenses</a>
          </div>
        </footer>
      </div>
    </>
  )
}
