import { useEffect, useMemo, useState } from "react";
import { api, type Storm, type TrackPoint } from "../api";
import MapView from "../components/MapView";
import { StormChip } from "./Overview";

export default function Archive() {
  const [q, setQ] = useState("");
  const [basin, setBasin] = useState("");
  const [storms, setStorms] = useState<Storm[]>([]);
  const [sel, setSel] = useState<Storm | null>(null);
  const [track, setTrack] = useState<TrackPoint[]>([]);

  useEffect(() => {
    const params = [q && `&q=${encodeURIComponent(q)}`, basin && `&basin=${basin}`].filter(Boolean).join("");
    api.storms(params).then((r) => setStorms(r.storms)).catch(() => setStorms([]));
  }, [q, basin]);

  async function open(s: Storm) {
    setSel(s);
    const d = await api.storm(s.sid);
    setTrack(d.track);
  }

  const center = useMemo<[number, number]>(() => {
    if (!track.length) return [15, 80];
    return [track[Math.floor(track.length / 2)].lat, track[Math.floor(track.length / 2)].lon];
  }, [track]);

  return (
    <div>
      <div className="kicker">Catalog</div>
      <h1>IBTrACS North Indian Ocean archive</h1>
      <p className="lede">Search named storms, inspect 6-hourly best-track, and read the derived track-pattern family.</p>
      <div className="toolbar">
        <input placeholder="Name or SID" value={q} onChange={(e) => setQ(e.target.value)} />
        <select value={basin} onChange={(e) => setBasin(e.target.value)}>
          <option value="">All basins</option>
          <option value="AS">Arabian Sea</option>
          <option value="BOB">Bay of Bengal</option>
        </select>
      </div>
      <div className="row cols-2">
        <div className="card" style={{ maxHeight: 560, overflow: "auto" }}>
          <table className="table">
            <thead>
              <tr>
                <th>Storm</th>
                <th>Year</th>
                <th>Peak</th>
                <th>Pattern</th>
              </tr>
            </thead>
            <tbody>
              {storms.map((s) => (
                <tr key={s.sid} onClick={() => void open(s)} style={{ cursor: "pointer" }}>
                  <td>
                    {s.name} <StormChip storm={s} />
                  </td>
                  <td>{s.season}</td>
                  <td>{s.peak_wind_kt} kt</td>
                  <td>{s.track_pattern}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="card">
          {sel ? (
            <>
              <h2>
                {sel.name} · {sel.season} · {sel.basin}
              </h2>
              <p className="lede">
                Genesis {sel.genesis_lat}N {sel.genesis_lon}E · {sel.n_points} six-hour fixes · min P{" "}
                {sel.min_pressure_hpa ?? "—"} hPa
              </p>
              <MapView tracks={[{ points: track, color: "#2ee6d6" }]} center={center} zoom={5} />
            </>
          ) : (
            <p className="lede">Select a storm to plot its track.</p>
          )}
        </div>
      </div>
    </div>
  );
}
