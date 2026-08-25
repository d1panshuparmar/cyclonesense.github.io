import { useState } from "react";
import { api, type IdentifyResult } from "../api";

export default function Identify() {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<IdentifyResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function run(file: File) {
    setBusy(true);
    setError(null);
    try {
      setResult(await api.identify(file));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function sample() {
    setBusy(true);
    setError(null);
    try {
      const s = await api.sampleImage();
      const blob = await (await fetch(s.image)).blob();
      await run(new File([blob], "sample.png", { type: "image/png" }));
    } catch (e) {
      setError(String(e));
      setBusy(false);
    }
  }

  return (
    <div>
      <div className="kicker">Module A</div>
      <h1>Identify and classify cyclone patterns</h1>
      <p className="lede">
        Upload an infrared satellite crop (INSAT-3D TIR or similar). The CNN reports cyclone presence, Dvorak-style
        cloud pattern, IMD intensity class, and estimated maximum sustained wind. Grad-CAM shows which pixels drove
        the detection.
      </p>
      <div className="row split">
        <div className="card">
          <label className="drop">
            <input
              type="file"
              accept="image/*"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) void run(f);
              }}
            />
            Drop an IR image or click to upload
          </label>
          <div className="toolbar" style={{ marginTop: 12 }}>
            <button className="btn ghost" disabled={busy} onClick={() => void sample()}>
              {busy ? "Running…" : "Try a synthetic IR sample"}
            </button>
          </div>
          {error && <div className="alert YELLOW">{error}</div>}
          {result && (
            <div className="preview-grid" style={{ marginTop: 12 }}>
              {result.preview && (
                <div>
                  <div className="label">Input (128×128)</div>
                  <img src={result.preview} alt="input" />
                </div>
              )}
              {result.gradcam && (
                <div>
                  <div className="label">Grad-CAM (presence)</div>
                  <img src={result.gradcam} alt="gradcam" />
                </div>
              )}
            </div>
          )}
        </div>
        <div className="card">
          {!result && <p className="lede">No frame analysed yet.</p>}
          {result && (
            <>
              <div className="stat">
                <div className="label">Detection</div>
                <div className="value">{result.is_cyclone ? "Cyclonic system" : "No cyclone"}</div>
                <div className="hint">Confidence {(result.cyclone_confidence * 100).toFixed(1)}%</div>
              </div>
              <h2 style={{ marginTop: 18 }}>Cloud pattern</h2>
              <p>
                <strong>{result.pattern.name}</strong> · Dvorak {result.pattern.dvorak}
              </p>
              <p className="lede">{result.pattern.description}</p>
              <div className="bar-list">
                {result.pattern.distribution.map((d) => (
                  <div className="bar-row" key={d.code}>
                    <span>{d.name}</span>
                    <div className="bar">
                      <span style={{ width: `${d.p * 100}%` }} />
                    </div>
                    <span>{(d.p * 100).toFixed(0)}</span>
                  </div>
                ))}
              </div>
              <h2 style={{ marginTop: 18 }}>IMD intensity</h2>
              <p>
                <span className="pill" style={{ background: result.category.color + "33", color: result.category.color }}>
                  {result.category.code}
                </span>{" "}
                {result.category.name}
              </p>
              <p className="mono">
                Estimated MSW {result.wind_kt} kt · {result.category.min_kt}–{result.category.max_kt} kt band
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
