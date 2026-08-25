import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type Stats, type Storm, type TrackPoint } from "../api";
import MapView from "../components/MapView";

export default function Overview() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [tracks, setTracks] = useState<{ points: TrackPoint[]; color: string }[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const s = await api.stats();
        const show = await api.showcase();
        const loaded: { points: TrackPoint[]; color: string }[] = [];
        for (const storm of show.storms.slice(0, 8)) {
          const d = await api.storm(storm.sid);
          loaded.push({ points: d.track, color: colorFor(storm.category) });
        }
        if (alive) {
          setStats(s);
          setTracks(loaded);
        }
      } catch (e) {
        if (alive) setErr("Backend not ready. Start the API and train the catalog (see README).");
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const maxCat = stats ? Math.max(...Object.values(stats.by_category), 1) : 1;

  return (
    <div>
      <div className="kicker">North Indian Ocean</div>
      <h1>Tropical cyclone pattern intelligence</h1>
      <p className="lede">
        Identify cyclone structure in infrared imagery, classify intensity on the IMD scale, and forecast
        72-hour track and wind using a track LSTM plus historical analogs from IBTrACS.
      </p>
      {err && <div className="alert YELLOW">{err}</div>}
      {stats && (
        <>
          <div className="row cols-4">
            <div className="card stat">
              <div className="label">Catalogued storms</div>
              <div className="value">{stats.n_storms}</div>
              <div className="hint">
                {stats.season_min}–{stats.season_max} · IBTrACS NI
              </div>
            </div>
            <div className="card stat">
              <div className="label">Arabian Sea / BoB</div>
              <div className="value">
                {stats.by_basin.AS ?? 0}
                <span style={{ color: "var(--muted)", fontSize: 16 }}> / {stats.by_basin.BOB ?? 0}</span>
              </div>
              <div className="hint">Basin split at 77.5°E</div>
            </div>
            <div className="card stat">
              <div className="label">Strongest in archive</div>
              <div className="value">{stats.strongest?.name}</div>
              <div className="hint">
                {stats.strongest?.peak_wind_kt} kt · {stats.strongest?.category} · {stats.strongest?.season}
              </div>
            </div>
            <div className="card stat">
              <div className="label">Recurving tracks</div>
              <div className="value">{stats.by_track_pattern.RECURVING ?? 0}</div>
              <div className="hint">Pattern family over NIO</div>
            </div>
          </div>
          <div className="row cols-2" style={{ marginTop: 16 }}>
            <div className="card">
              <h2>Recent intense tracks</h2>
              <MapView tracks={tracks} />
            </div>
            <div className="card">
              <h2>IMD category distribution</h2>
              <div className="bar-list">
                {Object.entries(stats.by_category)
                  .sort((a, b) => b[1] - a[1])
                  .map(([k, v]) => (
                    <div className="bar-row" key={k}>
                      <span>{k}</span>
                      <div className="bar">
                        <span style={{ width: `${(100 * v) / maxCat}%`, background: colorFor(k) }} />
                      </div>
                      <span>{v}</span>
                    </div>
                  ))}
              </div>
              <p className="lede" style={{ marginTop: 18, marginBottom: 0 }}>
                <Link to="/identify">Identify a satellite frame</Link> · <Link to="/predict">Run a 72-h forecast</Link>
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function colorFor(code: string) {
  const m: Record<string, string> = {
    LPA: "#94a3b8",
    D: "#38bdf8",
    DD: "#22d3ee",
    CS: "#facc15",
    SCS: "#fb923c",
    VSCS: "#f97316",
    ESCS: "#ef4444",
    SuCS: "#a855f7",
  };
  return m[code] || "#2ee6d6";
}

export function StormChip({ storm }: { storm: Storm }) {
  return (
    <span className="pill" style={{ background: colorFor(storm.category) + "33", color: colorFor(storm.category) }}>
      {storm.category}
    </span>
  );
}
