import { Circle, CircleMarker, MapContainer, Polyline, Popup, TileLayer } from "react-leaflet";
import type { TrackPoint } from "../api";

type Track = { points: TrackPoint[]; color?: string; dash?: string; weight?: number };

export default function MapView({
  tracks = [],
  cones = [],
  center = [15, 80],
  zoom = 5,
}: {
  tracks?: Track[];
  cones?: TrackPoint[];
  center?: [number, number];
  zoom?: number;
}) {
  return (
    <div className="map-wrap">
      <MapContainer center={center} zoom={zoom} style={{ height: "100%", width: "100%" }} scrollWheelZoom>
        <TileLayer
          attribution='&copy; OpenStreetMap &copy; CARTO'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        />
        {cones.map((p, i) =>
          p.uncertainty_km ? (
            <Circle
              key={`c-${i}`}
              center={[p.lat, p.lon]}
              radius={p.uncertainty_km * 1000}
              pathOptions={{ color: "#2ee6d6", weight: 0, fillOpacity: 0.06 }}
            />
          ) : null
        )}
        {tracks.map((t, i) => (
          <Polyline
            key={i}
            positions={t.points.map((p) => [p.lat, p.lon] as [number, number])}
            pathOptions={{ color: t.color || "#2ee6d6", weight: t.weight || 3, dashArray: t.dash }}
          />
        ))}
        {tracks.flatMap((t, ti) =>
          t.points.map((p, i) => (
            <CircleMarker
              key={`${ti}-${i}`}
              center={[p.lat, p.lon]}
              radius={i === t.points.length - 1 ? 7 : 3}
              pathOptions={{ color: p.color || t.color || "#2ee6d6", fillOpacity: 0.9, weight: 1 }}
            >
              <Popup>
                <div className="mono">
                  {p.time}
                  <br />
                  {p.lat.toFixed(2)}N {p.lon.toFixed(2)}E
                  <br />
                  {p.wind_kt ?? "—"} kt · {p.category}
                </div>
              </Popup>
            </CircleMarker>
          ))
        )}
      </MapContainer>
    </div>
  );
}
