export type Storm = {
  sid: string;
  name: string;
  season: number | null;
  start: string;
  end: string;
  peak_wind_kt: number;
  min_pressure_hpa: number | null;
  category: string;
  category_name: string;
  basin: string;
  track_pattern: string;
  n_points: number;
  genesis_lat: number;
  genesis_lon: number;
};

export type TrackPoint = {
  time: string;
  lat: number;
  lon: number;
  wind_kt: number | null;
  pres_hpa?: number | null;
  category: string;
  color: string;
  lead_hours?: number;
  uncertainty_km?: number;
  category_name?: string;
};

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export const api = {
  health: () => get<Record<string, unknown>>("/api/health"),
  taxonomy: () => get<{ imd_categories: Category[]; cloud_patterns: Pattern[]; track_patterns: { code: string; name: string }[] }>("/api/taxonomy"),
  stats: () => get<Stats>("/api/stats"),
  storms: (q = "") => get<{ storms: Storm[] }>(`/api/storms?limit=120${q}`),
  showcase: () => get<{ storms: Storm[] }>("/api/storms/showcase"),
  storm: (sid: string) => get<{ storm: Storm; track: TrackPoint[] }>(`/api/storms/${encodeURIComponent(sid)}`),
  predictStorm: (sid: string) => get<PredictStorm>(`/api/predict/${encodeURIComponent(sid)}`),
  sampleImage: () => get<{ image: string; label: Record<string, unknown> }>("/api/sample-image"),
  async identify(file: File) {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/api/identify", { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    return res.json() as Promise<IdentifyResult>;
  },
  async predict(obs: { time?: string; lat: number; lon: number; wind?: number }[], hours = 72) {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ observations: obs, hours, include_analogs: true }),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json() as Promise<{ lstm: Forecast; analog: { analogs: Analog[]; forecast: TrackPoint[] } }>;
  },
};

export type Category = {
  code: string;
  name: string;
  min_kt: number;
  max_kt: number;
  color: string;
};

export type Pattern = {
  code: string;
  name: string;
  dvorak: string;
  description: string;
};

export type Stats = {
  ready: boolean;
  n_storms: number;
  season_min: number;
  season_max: number;
  by_category: Record<string, number>;
  by_basin: Record<string, number>;
  by_track_pattern: Record<string, number>;
  by_season: { season: number; n: number }[];
  strongest: Storm;
};

export type IdentifyResult = {
  is_cyclone: boolean;
  cyclone_confidence: number;
  pattern: {
    code: string;
    name: string;
    dvorak: string;
    description: string;
    confidence: number;
    distribution: { code: string; name: string; p: number }[];
  };
  category: Category & { confidence: number };
  wind_kt: number;
  gradcam?: string | null;
  preview?: string;
  notes?: string;
};

export type Forecast = {
  horizon_hours: number;
  forecast: TrackPoint[];
  peak_forecast_wind_kt: number;
  peak_forecast_category: string;
  land_threat: { level: string; message: string; eta_hours?: number };
};

export type Analog = {
  sid: string;
  name: string;
  season: number;
  distance: number;
  peak_wind_kt: number;
};

export type PredictStorm = {
  storm: Storm;
  history: TrackPoint[];
  observed_future: TrackPoint[];
  lstm: Forecast;
  analog: { analogs: Analog[]; forecast: TrackPoint[] };
};
