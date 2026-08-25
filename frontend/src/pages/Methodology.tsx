export default function Methodology() {
  return (
    <div className="method">
      <div className="kicker">Design</div>
      <h1>How CycloneSense works</h1>
      <p className="lede">
        Three coupled tasks match the problem statement: identification of cyclonic cloud organization, classification
        into operational IMD intensity and Dvorak-style patterns, and prediction of track and intensity over the North
        Indian Ocean.
      </p>
      <div className="row cols-3">
        <div className="card">
          <h2>1. Identify</h2>
          <p className="lede">
            A compact multi-task CNN reads 128×128 infrared crops. The presence head is a cyclone / non-cyclone
            detector. Grad-CAM highlights the organised cold-cloud core that triggered the decision.
          </p>
        </div>
        <div className="card">
          <h2>2. Classify</h2>
          <p className="lede">
            Shared visual features feed a five-class cloud-pattern head (curved band, shear, CDO, embedded centre, eye)
            and an IMD intensity head (D through SuCS) plus a wind-speed regressor in knots.
          </p>
        </div>
        <div className="card">
          <h2>3. Predict</h2>
          <p className="lede">
            A 2-layer LSTM encoder–decoder is trained on IBTrACS NI 6-hourly sequences (1980–2017 train, 2018–2021
            val, 2022+ test). It emits 12 steps of Δlat, Δlon, Δwind (72 h). Analogs provide a transparent ensemble.
          </p>
        </div>
      </div>
      <div className="card" style={{ marginTop: 16 }}>
        <h2>Data</h2>
        <ol>
          <li>
            <strong>IBTrACS v04r01</strong> North Indian basin best-track, interpolated to 6-hourly fixes. Wind prefers
            New Delhi (IMD) then WMO then JTWC.
          </li>
          <li>
            <strong>INSAT-3D/3DR TIR</strong> (MOSDAC) is the operational image source. The bundled CNN is pretrained on
            physically motivated IR pattern synthesis so the pipeline runs without a MOSDAC login; retrain with{" "}
            <code>python -m backend.ml.train_images</code> on real TIR crops labelled from IMD best-track.
          </li>
          <li>
            Intensity labels follow IMD 2015 thresholds: D 17–27 kt, DD 28–33, CS 34–47, SCS 48–63, VSCS 64–89, ESCS
            90–119, SuCS ≥120.
          </li>
        </ol>
      </div>
      <div className="card" style={{ marginTop: 16 }}>
        <h2>Why this is not a black box</h2>
        <p className="lede">
          Pattern classes map onto Dvorak T-numbers used at RSMC New Delhi. Track families (westward, northwest,
          recurving, looping) are derived from the observed path. Analog storms named in the forecast panel are real
          IBTrACS events with similar location, motion, and intensity — the same reasoning a duty forecaster uses.
        </p>
      </div>
    </div>
  );
}
