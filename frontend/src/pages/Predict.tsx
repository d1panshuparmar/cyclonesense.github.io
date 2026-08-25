import { useEffect, useState } from "react";
import { api, type PredictStorm, type Storm } from "../api";
import MapView from "../components/MapView";

export default function Predict() {
  const [storms, setStorms] = useState<Storm[]>([]);
  const [sid, setSid] = useState("");
  const [result, setResult] = useState<PredictStorm | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.storms("&category=VSCS")
      .then((r) => {
        const list = r.storms.length ? r.storms : [];
        return list.length ? r : api.storms();
      })
      .then((r) => {
        setStorms(r.storms);
        if (r.storms[0]) setSid(r.storms[0].sid);
      })
      .catch(() => setErr("Could not load storms."));
  }, []);

  async function run() {
    if (!sid) return;
    setBusy(true);
    setErr(null);
    try {
      setResult(await api.predictStorm(sid));
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  const threat = result?.lstm.land_threat;

  return (
    <div>
      <div className="kicker">Module B</div>
      <h1>72-hour track and intensity forecast</h1>
      <p className="lede">
        An encoder–decoder LSTM trained on IBTrACS 6-hourly North Indian Ocean tracks predicts displacement and wind
        change. A k-nearest analog ensemble from historical storms is shown as an interpretable baseline. Verification
        uses the withheld remainder of the selected storm.
      </p>
      <div className="toolbar">
        <select value={sid} onChange={(e) => setSid(e.target.value)}>
          {storms.map((s) => (
            <option key={s.sid} value={s.sid}>
              {s.season} {s.name} · {s.category} · {s.peak_wind_kt} kt
            </option>
          ))}
        </select>
        <button className="btn" disabled={busy || !sid} onClick={() => void run()}>
          {busy ? "Forecasting…" : "Forecast from early track"}
        </button>
      </div>
      {err && <div className="alert YELLOW">{err}</div>}
      {result && (
        <>
          {threat && <div className={`alert ${threat.level}`}>{threat.message}</div>}
          <div className="row cols-2" style={{ marginTop: 16 }}>
            <div className="card">
              <h2>
                {result.storm.name} ({result.storm.season}) · {result.storm.track_pattern.toLowerCase()} track
              </h2>
              <MapView
                center={[
                  result.history[result.history.length - 1]?.lat ?? 15,
                  result.history[result.history.length - 1]?.lon ?? 80,
                ]}
                tracks={[
                  { points: result.history, color: "#e7eef6", weight: 3 },
                  { points: result.observed_future, color: "#64748b", dash: "4 6", weight: 2 },
                  { points: result.lstm.forecast, color: "#2ee6d6", weight: 3 },
                  { points: result.analog.forecast, color: "#f5b942", dash: "6 4", weight: 2 },
                ]}
                cones={result.lstm.forecast}
              />
              <div className="legend">
                <span>
                  <i style={{ background: "#e7eef6" }} /> Observed (input)
                </span>
                <span>
                  <i style={{ background: "#64748b" }} /> Observed (held out)
                </span>
                <span>
                  <i style={{ background: "#2ee6d6" }} /> LSTM forecast
                </span>
                <span>
                  <i style={{ background: "#f5b942" }} /> Analog ensemble
                </span>
              </div>
            </div>
            <div className="card">
              <h2>Lead-time guidance</h2>
              <table className="table">
                <thead>
                  <tr>
                    <th>Lead</th>
                    <th>LSTM</th>
                    <th>Wind</th>
                    <th>IMD</th>
                  </tr>
                </thead>
                <tbody>
                  {result.lstm.forecast
                    .filter((p) => p.lead_hours && p.lead_hours % 12 === 0)
                    .map((p) => (
                      <tr key={p.lead_hours}>
                        <td className="mono">H+{p.lead_hours}</td>
                        <td className="mono">
                          {p.lat.toFixed(1)}N {p.lon.toFixed(1)}E
                        </td>
                        <td>{p.wind_kt} kt</td>
                        <td>{p.category}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
              <h2 style={{ marginTop: 18 }}>Nearest historical analogs</h2>
              <table className="table">
                <thead>
                  <tr>
                    <th>Storm</th>
                    <th>Year</th>
                    <th>Peak</th>
                  </tr>
                </thead>
                <tbody>
                  {result.analog.analogs.map((a) => (
                    <tr key={a.sid + a.distance}>
                      <td>{a.name}</td>
                      <td>{a.season}</td>
                      <td>{a.peak_wind_kt} kt</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
