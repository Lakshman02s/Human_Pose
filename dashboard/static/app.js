(function () {
  const root = document.getElementById("root");
  const state = {
    payload: null,
    activeTab: "overview",
    selectedVideoId: null,
  };

  const COLORS = {
    cyan: "#37d0ff",
    mint: "#73f0b5",
    ember: "#ff8f5a",
    plum: "#8f8bff",
    red: "#ff7a7a",
  };

  function fmt(value, digits = 2) {
    return Number(value || 0).toFixed(digits);
  }

  function fmtPct(value, digits = 1) {
    return `${Number((value || 0) * 100).toFixed(digits)}%`;
  }

  function titleCase(value) {
    return String(value || "")
      .replace(/[_-]+/g, " ")
      .replace(/\b\w/g, (char) => char.toUpperCase());
  }

  function statCard(label, value, hint, tone) {
    return `
      <div class="metric-panel rounded-[26px] p-5">
        <div class="text-[11px] uppercase tracking-[0.28em] text-slate-400">${label}</div>
        <div class="metric-value tone-${tone} mt-3 text-4xl font-semibold">${value}</div>
        <div class="mt-2 text-sm text-slate-400">${hint}</div>
      </div>
    `;
  }

  function section(eyebrow, title, subtitle, body, actions) {
    return `
      <section class="glass-panel rounded-[30px] p-6 lg:p-7">
        <div class="mb-6 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <div class="text-[11px] uppercase tracking-[0.32em] text-cyan-300/80">${eyebrow}</div>
            <div class="mt-3 text-3xl font-semibold text-white">${title}</div>
            <div class="mt-2 max-w-4xl text-sm text-slate-400">${subtitle}</div>
          </div>
          ${actions || ""}
        </div>
        ${body}
      </section>
    `;
  }

  function table(headers, rows) {
    if (!rows.length) {
      return `<div class="rounded-[22px] border border-dashed border-white/10 p-4 text-slate-400">No data available.</div>`;
    }
    return `
      <div class="overflow-x-auto rounded-[24px] border border-white/8">
        <table class="min-w-full text-left text-sm">
          <thead class="bg-slate-950/45 text-slate-300">
            <tr>${headers.map((header) => `<th class="px-4 py-3 font-medium">${header.label}</th>`).join("")}</tr>
          </thead>
          <tbody class="divide-y divide-white/6 bg-slate-950/20">
            ${rows
              .map(
                (row) => `
              <tr>
                ${headers.map((header) => `<td class="px-4 py-3 text-slate-200">${row[header.key] ?? "-"}</td>`).join("")}
              </tr>
            `
              )
              .join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  function renderLoading() {
    root.innerHTML = `<div class="grid-shell flex min-h-screen items-center justify-center text-xl text-slate-300">Loading enterprise benchmarking suite...</div>`;
  }

  function renderError(message) {
    root.innerHTML = `<div class="grid-shell flex min-h-screen items-center justify-center px-6 text-center text-red-300">${message}</div>`;
  }

  function getSelectedVideoRun() {
    const runs = state.payload.video_validations || [];
    if (!state.selectedVideoId || !runs.find((run) => run.id === state.selectedVideoId)) {
      state.selectedVideoId = runs[0] ? runs[0].id : null;
    }
    return runs.find((run) => run.id === state.selectedVideoId) || null;
  }

  function renderOverview() {
    const payload = state.payload;
    const overview = payload.overview || {};
    const recommendations = payload.recommendations || [];
    const recommendation = recommendations[0];
    const modelProfiles = payload.model_profiles || [];
    const official = payload.official_coco_baselines || { models: [] };

    const usageRows = modelProfiles.map((profile) => `
      <div class="mini-metric rounded-[24px] p-4">
        <div class="flex items-center justify-between gap-3">
          <div class="text-lg font-semibold text-white">${String(profile.model || "").toUpperCase()}</div>
          <span class="status-pill ${profile.model === "rtmpose" ? "status-positive" : "status-neutral"}">${profile.success_ratio >= 0.7 ? "Ready" : "Review"}</span>
        </div>
        <div class="mt-4 grid grid-cols-2 gap-3 text-sm">
          <div><div class="text-slate-500">Avg FPS</div><div class="mt-1 font-semibold text-cyan-300">${fmt(profile.avg_fps)}</div></div>
          <div><div class="text-slate-500">Latency</div><div class="mt-1 font-semibold text-amber-300">${fmt(profile.avg_latency_ms)} ms</div></div>
          <div><div class="text-slate-500">Success</div><div class="mt-1 font-semibold text-mint-300">${fmtPct(profile.success_ratio, 0)}</div></div>
          <div><div class="text-slate-500">COCO mAP</div><div class="mt-1 font-semibold text-plum-300">${fmt(profile.official_map, 3)}</div></div>
        </div>
      </div>
    `).join("");

    return `
      <div class="space-y-6">
        <div class="grid gap-6 xl:grid-cols-[1.25fr_0.95fr]">
          <div class="glass-panel hero-shell rounded-[34px] p-7">
            <div class="hero-grid"></div>
            <div class="relative z-10">
              <div class="text-[11px] uppercase tracking-[0.32em] text-cyan-300/80">AI Benchmarking Suite</div>
              <h1 class="mt-4 max-w-4xl text-5xl font-semibold leading-tight text-white lg:text-6xl">Human Pose Detection Benchmark Platform</h1>
              <p class="mt-4 max-w-3xl text-base text-slate-300">Enterprise-style observability for RTMPose and ViTPose across recorded validation clips, live RTSP feeds, and deployment readiness reviews.</p>
              <div class="mt-8 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                ${statCard("Recommended Model", overview.recommended_model || "-", "Current deployment-ready winner", "cyan")}
                ${statCard("Total Runs", overview.total_runs || 0, "Stored benchmark and validation runs", "mint")}
                ${statCard("Average FPS", fmt(overview.avg_fps || 0), "Cross-run throughput signal", "cyan")}
                ${statCard("Average Latency", `${fmt(overview.avg_latency_ms || 0)} ms`, "Cross-run end-to-end mean", "ember")}
              </div>
            </div>
          </div>
          <div class="glass-panel rounded-[34px] p-7">
            <div class="flex items-center justify-between">
              <div>
                <div class="text-[11px] uppercase tracking-[0.28em] text-slate-400">Recommendation Engine</div>
                <div class="mt-3 text-2xl font-semibold text-white">Deployment Guidance</div>
              </div>
              <span class="status-pill status-positive">${overview.live_status || "Pending"}</span>
            </div>
            <div class="mt-6 rounded-[24px] border border-white/8 bg-slate-950/30 p-5">
              <div class="text-sm font-medium text-slate-300">Primary Recommendation</div>
              <div class="mt-3 text-2xl font-semibold text-white">${recommendation ? recommendation.title : "Recommendation pending"}</div>
              <p class="mt-3 text-sm leading-6 text-slate-400">${recommendation ? recommendation.summary : "Run the benchmark suite to generate recommendations."}</p>
            </div>
            <div class="mt-5 grid gap-4 sm:grid-cols-2">
              <div class="mini-metric rounded-[22px] p-4"><div class="text-xs uppercase tracking-[0.24em] text-slate-400">Live Stream Status</div><div class="mt-3 text-2xl font-semibold text-mint-300">${overview.live_status || "Pending"}</div><div class="mt-2 text-sm text-slate-400">Real camera feed validation</div></div>
              <div class="mini-metric rounded-[22px] p-4"><div class="text-xs uppercase tracking-[0.24em] text-slate-400">System Status</div><div class="mt-3 text-2xl font-semibold text-cyan-300">${overview.system_status || "Stable"}</div><div class="mt-2 text-sm text-slate-400">CPU-focused benchmark environment</div></div>
              <div class="mini-metric rounded-[22px] p-4"><div class="text-xs uppercase tracking-[0.24em] text-slate-400">Recorded Feeds</div><div class="mt-3 text-2xl font-semibold text-plum-300">${overview.recorded_runs || 0}</div><div class="mt-2 text-sm text-slate-400">Offline validation assets</div></div>
              <div class="mini-metric rounded-[22px] p-4"><div class="text-xs uppercase tracking-[0.24em] text-slate-400">Live Feeds</div><div class="mt-3 text-2xl font-semibold text-amber-300">${overview.live_runs || 0}</div><div class="mt-2 text-sm text-slate-400">RTSP sessions captured</div></div>
            </div>
          </div>
        </div>
        <div class="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
          ${section(
            "Model Comparison",
            "Throughput, Latency, and Resource Trade-Offs",
            "The core decision layer compares benchmark speed, system cost, and field stability in one view.",
            `
              <div class="grid gap-6 lg:grid-cols-2">
                <div class="chart-panel rounded-[24px] p-4">
                  <div class="chart-title">Grouped Benchmark Comparison</div>
                  <div class="chart-subtitle">FPS, latency, CPU, and RAM side-by-side.</div>
                  <div class="mt-4 h-[320px]"><canvas id="overviewBarChart"></canvas></div>
                </div>
                <div class="chart-panel rounded-[24px] p-4">
                  <div class="chart-title">Deployment Readiness Radar</div>
                  <div class="chart-subtitle">Speed, robustness, accuracy, deployability, and stability.</div>
                  <div class="mt-4 h-[320px]"><canvas id="overviewRadarChart"></canvas></div>
                </div>
              </div>
            `
          )}
          ${section(
            "Usage Distribution",
            "Adoption and Runtime Mix",
            "Quick-read visuals for model utilization and live success status.",
            `
              <div class="grid gap-6 lg:grid-cols-2">
                <div class="chart-panel rounded-[24px] p-4">
                  <div class="chart-title">Model Usage Split</div>
                  <div class="mt-4 h-[240px]"><canvas id="overviewPieChart"></canvas></div>
                </div>
                <div class="space-y-4">${usageRows}</div>
              </div>
            `
          )}
        </div>
      </div>
    `;
  }

  function renderCompare() {
    const benchmarkRows = (state.payload.latest_benchmark || []).map((row) => ({
      model: String(row.model || "").toUpperCase(),
      fps: fmt(row.fps_mean),
      latency: `${fmt(row.latency_ms_mean)} ms`,
      cpu: `${fmt(row.cpu_percent_mean, 1)} %`,
      ram: `${fmt(row.ram_percent_mean, 1)} %`,
      detector: `${fmt(row.detector_ms_mean)} ms`,
      pose: `${fmt(row.pose_ms_mean)} ms`,
    }));

    const officialRows = (state.payload.official_coco_baselines?.models || []).map((row) => ({
      model: String(row.model || "").toUpperCase(),
      variant: row.variant,
      map: fmt(row.mAP, 3),
      map50: fmt(row.mAP50, 3),
      map75: fmt(row.mAP75, 3),
      recall: fmt(row.recall, 3),
    }));

    return `
      <div class="space-y-6">
        ${section(
          "Model Benchmark",
          "RTMPose vs ViTPose",
          "Use this section to answer which model is faster, which is heavier, and which is more practical to standardize on.",
          table(
            [
              { key: "model", label: "Model" },
              { key: "fps", label: "Mean FPS" },
              { key: "latency", label: "Mean Latency" },
              { key: "cpu", label: "CPU %" },
              { key: "ram", label: "RAM %" },
              { key: "detector", label: "Detector Time" },
              { key: "pose", label: "Pose Time" },
            ],
            benchmarkRows
          )
        )}
        <div class="grid gap-6 xl:grid-cols-2">
          ${section(
            "COCO Baseline",
            "Official Accuracy Reference",
            "Official COCO 17-keypoint benchmark values used as sanity-check baseline.",
            table(
              [
                { key: "model", label: "Model" },
                { key: "variant", label: "Variant" },
                { key: "map", label: "mAP" },
                { key: "map50", label: "mAP50" },
                { key: "map75", label: "mAP75" },
                { key: "recall", label: "Recall" },
              ],
              officialRows
            )
          )}
          ${section(
            "Distribution Views",
            "Latency and Confidence Spread",
            "These charts show whether the models behave consistently or have unstable long-tail behavior.",
            `
              <div class="grid gap-6">
                <div class="chart-panel rounded-[24px] p-4">
                  <div class="chart-title">Latency Distribution</div>
                  <div class="mt-4 h-[260px]"><canvas id="compareLatencyChart"></canvas></div>
                </div>
                <div class="chart-panel rounded-[24px] p-4">
                  <div class="chart-title">Confidence Distribution</div>
                  <div class="mt-4 h-[260px]"><canvas id="compareConfidenceChart"></canvas></div>
                </div>
              </div>
            `
          )}
        </div>
      </div>
    `;
  }

  function renderVideos() {
    const runs = state.payload.video_validations || [];
    const selected = getSelectedVideoRun();
    const cards = runs
      .map(
        (run) => `
      <button data-video-id="${run.id}" class="validation-card text-left ${selected && selected.id === run.id ? "active" : ""}">
        <div class="thumb-shell">
          ${
            run.thumbnail
              ? `<img src="/artifacts/${run.thumbnail}" alt="${run.video_name}" class="thumb-image" />`
              : `<div class="thumb-fallback">No Preview</div>`
          }
          <span class="status-pill overlay-pill ${run.model === "rtmpose" ? "status-positive" : "status-neutral"}">${String(run.model).toUpperCase()}</span>
        </div>
        <div class="mt-4">
          <div class="text-lg font-semibold text-white">${run.video_name}</div>
          <div class="mt-1 text-sm text-slate-400">${titleCase(run.run_summary.source_type)} · ${titleCase(run.model)}</div>
          <div class="mt-4 grid grid-cols-2 gap-3 text-sm">
            <div><div class="text-slate-500">FPS</div><div class="mt-1 font-semibold text-cyan-300">${fmt(run.metrics.avg_fps)}</div></div>
            <div><div class="text-slate-500">Latency</div><div class="mt-1 font-semibold text-amber-300">${fmt(run.metrics.avg_latency_ms)} ms</div></div>
            <div><div class="text-slate-500">People</div><div class="mt-1 font-semibold text-mint-300">${fmt(run.metrics.avg_people_count)}</div></div>
            <div><div class="text-slate-500">Confidence</div><div class="mt-1 font-semibold text-plum-300">${fmt(run.failure_metrics.avg_keypoint_confidence, 3)}</div></div>
          </div>
        </div>
      </button>`
      )
      .join("");

    return `
      ${section(
        "Video Validation",
        "Recorded Footage Review",
        "Each tested input video is connected to preview, performance summary, and failure indicators.",
        `
          <div class="grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
            <div class="grid gap-4 md:grid-cols-2">${cards || `<div class="text-slate-400">No recorded validation runs found.</div>`}</div>
            <div class="glass-subpanel rounded-[26px] p-5">
              ${
                selected
                  ? `
                <div class="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <div class="text-[11px] uppercase tracking-[0.28em] text-cyan-300/80">Selected Validation</div>
                    <div class="mt-2 text-2xl font-semibold text-white">${selected.video_name}</div>
                    <div class="mt-1 text-sm text-slate-400">${selected.selection_label}</div>
                  </div>
                  <span class="status-pill ${selected.model === "rtmpose" ? "status-positive" : "status-neutral"}">${String(selected.model).toUpperCase()}</span>
                </div>
                <div class="mt-5 overflow-hidden rounded-[24px] border border-white/8">
                  <video src="${selected.output_video_web ? `/artifacts/${selected.output_video_web}` : `/artifacts/${selected.output_video}`}" controls class="w-full bg-slate-950" preload="metadata"></video>
                </div>
                <div class="mt-5 grid gap-4 lg:grid-cols-2">
                  <div class="chart-panel rounded-[22px] p-4">
                    <div class="chart-title">FPS Trend</div>
                    <div class="mt-3 h-[220px]"><canvas id="videoFpsChart"></canvas></div>
                  </div>
                  <div class="chart-panel rounded-[22px] p-4">
                    <div class="chart-title">Latency Trend</div>
                    <div class="mt-3 h-[220px]"><canvas id="videoLatencyChart"></canvas></div>
                  </div>
                </div>
              `
                  : `<div class="rounded-[22px] border border-dashed border-white/10 p-6 text-slate-400">Select a video run to inspect.</div>`
              }
            </div>
          </div>
        `
      )}
    `;
  }

  function renderLive() {
    const liveRuns = state.payload.live_runs || [];
    const aggregate = state.payload.rtsp_aggregate || {};
    const items = liveRuns
      .map(
        (run) => `
      <div class="stream-card rounded-[26px] p-5">
        <div class="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div class="text-lg font-semibold text-white">${run.selection_label}</div>
            <div class="mt-1 text-sm text-slate-400">${String(run.stream_metrics.codec || "unknown").toUpperCase()} · ${titleCase(run.model)}</div>
          </div>
          <span class="status-pill ${
            run.stream_quality === "critical" ? "status-danger" : run.stream_quality === "degraded" ? "status-warning" : "status-positive"
          }">${titleCase(run.stream_quality)}</span>
        </div>
        <div class="mt-5 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <div class="mini-metric rounded-[20px] p-4"><div class="text-xs uppercase tracking-[0.22em] text-slate-500">Live FPS</div><div class="mt-2 text-2xl font-semibold text-cyan-300">${fmt(run.stream_metrics.processed_fps)}</div></div>
          <div class="mini-metric rounded-[20px] p-4"><div class="text-xs uppercase tracking-[0.22em] text-slate-500">Dropped</div><div class="mt-2 text-2xl font-semibold text-amber-300">${fmtPct(run.stream_metrics.dropped_frame_ratio, 0)}</div></div>
          <div class="mini-metric rounded-[20px] p-4"><div class="text-xs uppercase tracking-[0.22em] text-slate-500">Jitter</div><div class="mt-2 text-2xl font-semibold text-plum-300">${fmt(run.stream_metrics.network_jitter_ms)} ms</div></div>
          <div class="mini-metric rounded-[20px] p-4"><div class="text-xs uppercase tracking-[0.22em] text-slate-500">Latency</div><div class="mt-2 text-2xl font-semibold text-mint-300">${fmt(run.stream_metrics.stream_latency_ms)} ms</div></div>
        </div>
        <div class="mt-5 grid gap-4 lg:grid-cols-2">
          <div class="overflow-hidden rounded-[20px] border border-white/8">
            <video src="${run.output_video_web ? `/artifacts/${run.output_video_web}` : `/artifacts/${run.output_video}`}" controls class="w-full bg-slate-950" preload="metadata"></video>
          </div>
          <div class="chart-panel rounded-[20px] p-4">
            <div class="chart-title">Run Metadata</div>
            <div class="mt-4 space-y-3 text-sm text-slate-300">
              <div>Processed FPS: <span class="text-cyan-300">${fmt(run.stream_metrics.processed_fps)}</span></div>
              <div>Source FPS: <span class="text-white">${fmt(run.stream_metrics.stream_fps)}</span></div>
              <div>Dropped Frame Ratio: <span class="text-amber-300">${fmtPct(run.stream_metrics.dropped_frame_ratio, 0)}</span></div>
              <div>Codec: <span class="text-white">${String(run.stream_metrics.codec || "unknown").toUpperCase()}</span></div>
              <div>RTSP Mode: <span class="text-white">${titleCase(run.run_summary.rtsp_mode || "smooth")}</span></div>
            </div>
          </div>
        </div>
      </div>`
      )
      .join("");

    return `
      <div class="space-y-6">
        <div class="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          ${statCard("Healthy Streams", aggregate.healthy_streams || 0, "Sessions with stable stream quality", "mint")}
          ${statCard("Degraded Streams", aggregate.degraded_streams || 0, "Sessions with quality concerns", "ember")}
          ${statCard("Avg Stream Latency", `${fmt(aggregate.avg_stream_latency_ms || 0)} ms`, "Across RTSP captures", "cyan")}
          ${statCard("Dropped Frame Ratio", fmtPct(aggregate.avg_dropped_ratio || 0, 0), "Approximate live capture loss", "plum")}
        </div>
        ${section(
          "Live Stream Analytics",
          "RTSP Monitoring and Stream Health",
          "Dedicated view for stream quality, dropped frames, processing speed, and stability over real camera feeds.",
          `
            <div class="grid gap-6">
              <div class="chart-panel rounded-[24px] p-4">
                <div class="chart-title">Stream Quality Distribution</div>
                <div class="chart-subtitle">Stable vs degraded vs critical sessions.</div>
                <div class="mt-4 h-[260px]"><canvas id="liveQualityChart"></canvas></div>
              </div>
              <div class="grid gap-4">${items || `<div class="rounded-[24px] border border-dashed border-white/10 p-6 text-slate-400">No RTSP runs stored yet.</div>`}</div>
            </div>
          `
        )}
      </div>
    `;
  }

  function renderFailures() {
    const failureRows = state.payload.failure_rows || [];
    const distributions = state.payload.distributions || {};
    const heatRows = ["rtmpose", "vitpose"]
      .map((model) => {
        const rows = failureRows.filter((row) => row.model === model);
        return `
          <div>
            <div class="mb-2 text-sm font-medium text-slate-300">${String(model).toUpperCase()}</div>
            <div class="grid grid-cols-4 gap-3">
              ${["distance", "occlusion", "crowding", "angle"]
                .map((category) => {
                  const avg = rows.length ? rows.reduce((sum, row) => sum + Number(row[category] || 0), 0) / rows.length : 0;
                  const intensity = Math.min(1, avg / 120);
                  return `
                    <div class="heat-cell rounded-[18px] p-3" style="background:linear-gradient(180deg, rgba(55,208,255,${0.08 + intensity * 0.42}), rgba(255,143,90,${0.05 + intensity * 0.28}))">
                      <div class="text-[11px] uppercase tracking-[0.2em] text-slate-300">${category}</div>
                      <div class="mt-2 text-2xl font-semibold text-white">${fmt(avg)}</div>
                    </div>
                  `;
                })
                .join("")}
            </div>
          </div>
        `;
      })
      .join("");

    return `
      ${section(
        "Failure Analysis",
        "Where The Models Break",
        "A compact observability layer for far-distance misses, occlusion, crowding, angle failure, and stream instability.",
        `
          <div class="grid gap-6">
            <div class="space-y-6">
              <div class="chart-panel rounded-[24px] p-4">
                <div class="chart-title">Run Severity Mix</div>
                <div class="chart-subtitle">Success vs warning vs critical validation outcomes.</div>
                <div class="mt-4 h-[260px]"><canvas id="failureSeverityChart"></canvas></div>
              </div>
              <div class="chart-panel rounded-[24px] p-4">
                <div class="chart-title">Failure Condition Heatmap</div>
                <div class="chart-subtitle">Average failure load by model and condition.</div>
                <div class="mt-4 space-y-3">${heatRows}</div>
              </div>
            </div>
            <div class="grid gap-6 xl:grid-cols-2">
              <div class="chart-panel rounded-[24px] p-4">
                <div class="chart-title">Latency Distribution</div>
                <div class="mt-4 h-[320px]"><canvas id="failureLatencyChart"></canvas></div>
              </div>
              <div class="chart-panel rounded-[24px] p-4">
                <div class="chart-title">Confidence Distribution</div>
                <div class="mt-4 h-[320px]"><canvas id="failureConfidenceChart"></canvas></div>
              </div>
            </div>
          </div>
        `
      )}
    `;
  }

  function renderHistory() {
    const items = (state.payload.benchmark_history || [])
      .slice()
      .reverse()
      .map(
        (entry) => `
      <div class="timeline-row">
        <div class="timeline-dot"></div>
        <div class="timeline-content rounded-[22px] p-4">
          <div class="flex flex-wrap items-center justify-between gap-3">
            <div class="text-base font-semibold text-white">${entry.timestamp}</div>
            <span class="status-pill status-neutral">${String(entry.device || "cpu").toUpperCase()}</span>
          </div>
          <div class="mt-2 text-sm text-slate-400">Source: ${entry.source} · Frames: ${entry.max_frames}</div>
          <div class="mt-3 flex flex-wrap gap-2">${(entry.models || []).map((model) => `<span class="chip">${String(model).toUpperCase()}</span>`).join("")}</div>
        </div>
      </div>
    `
      )
      .join("");

    const recs = (state.payload.recommendations || [])
      .map(
        (item) => `
      <div class="recommend-card rounded-[24px] p-5">
        <div class="flex items-center justify-between gap-3">
          <div class="text-lg font-semibold text-white">${item.title}</div>
          <span class="status-pill ${
            item.severity === "positive" ? "status-positive" : item.severity === "warning" ? "status-warning" : "status-neutral"
          }">${titleCase(item.severity)}</span>
        </div>
        <p class="mt-3 text-sm leading-6 text-slate-300">${item.summary}</p>
      </div>
    `
      )
      .join("");

    return `
      ${section(
        "Session Management",
        "Experiments, Recommendations, and Traceability",
        "Searchable benchmark sessions and AI-style recommendations for architecture and deployment decisions.",
        `
          <div class="grid gap-6 xl:grid-cols-[0.9fr_1.1fr]">
            <div class="space-y-4">${recs}</div>
            <div class="chart-panel rounded-[24px] p-4">
              <div class="chart-title">Benchmark History Timeline</div>
              <div class="chart-subtitle">Timestamped experiments captured through benchmark.py.</div>
              <div class="mt-4 space-y-4">${items || `<div class="rounded-[22px] border border-dashed border-white/10 p-4 text-slate-400">No benchmark history stored yet.</div>`}</div>
            </div>
          </div>
        `
      )}
    `;
  }

  function renderApp() {
    const tabs = [
      { key: "overview", label: "Executive Overview" },
      { key: "compare", label: "Model Comparison" },
      { key: "videos", label: "Video Validation" },
      { key: "live", label: "Live Streams" },
      { key: "failures", label: "Failure Analysis" },
      { key: "history", label: "Benchmark Sessions" },
    ];

    const content = {
      overview: renderOverview(),
      compare: renderCompare(),
      videos: renderVideos(),
      live: renderLive(),
      failures: renderFailures(),
      history: renderHistory(),
    }[state.activeTab];

    root.innerHTML = `
      <div class="grid-shell min-h-screen px-5 py-5 md:px-8 md:py-8">
        <div class="mx-auto max-w-[1680px] space-y-6">
          <div class="nav-shell flex flex-col gap-4 rounded-[30px] p-5 xl:flex-row xl:items-center xl:justify-between">
            <div class="flex flex-wrap items-center gap-4">
              <div class="logo-orb"></div>
              <div>
                <div class="text-xs uppercase tracking-[0.3em] text-cyan-300/80">Enterprise Pose Observability</div>
                <div class="mt-1 text-2xl font-semibold text-white">Benchmarking Suite</div>
              </div>
            </div>
            <div class="flex flex-wrap gap-3">
              ${tabs
                .map(
                  (tab) => `<button data-tab="${tab.key}" class="tab-chip ${state.activeTab === tab.key ? "active" : ""}">${tab.label}</button>`
                )
                .join("")}
              <a href="/report" class="tab-chip accent-link">Open Client Report</a>
            </div>
          </div>
          <div class="space-y-6">${content}</div>
        </div>
      </div>
    `;

    document.querySelectorAll("[data-tab]").forEach((button) => {
      button.addEventListener("click", () => {
        state.activeTab = button.dataset.tab;
        renderApp();
        renderCharts();
      });
    });

    document.querySelectorAll("[data-video-id]").forEach((button) => {
      button.addEventListener("click", () => {
        state.selectedVideoId = button.dataset.videoId;
        state.activeTab = "videos";
        renderApp();
        renderCharts();
      });
    });
  }

  function buildHistogram(values, step) {
    if (!values || !values.length) return [];
    const max = Math.max.apply(null, values);
    const rows = [];
    for (let start = 0; start <= max + step; start += step) {
      const end = start + step;
      rows.push({
        label: `${Math.round(start)}-${Math.round(end)}`,
        count: values.filter((value) => value >= start && value < end).length,
      });
    }
    return rows;
  }

  function renderCharts() {
    const payload = state.payload;
    if (!payload) return;

    const benchmark = payload.latest_benchmark || [];
    const profiles = payload.model_profiles || [];
    const official = payload.official_coco_baselines?.models || [];

    const overviewBar = document.getElementById("overviewBarChart");
    if (overviewBar) {
      new Chart(overviewBar, {
        type: "bar",
        data: {
          labels: profiles.map((item) => String(item.model || "").toUpperCase()),
          datasets: [
            { label: "FPS", data: profiles.map((item) => Number(item.benchmark_fps || item.avg_fps || 0)), backgroundColor: COLORS.cyan },
            { label: "Latency (ms)", data: profiles.map((item) => Number(item.benchmark_latency_ms || item.avg_latency_ms || 0)), backgroundColor: COLORS.ember },
            { label: "CPU %", data: profiles.map((item) => Number(item.avg_cpu_percent || 0)), backgroundColor: COLORS.mint },
            { label: "RAM %", data: profiles.map((item) => Number(item.avg_ram_percent || 0)), backgroundColor: COLORS.plum },
          ],
        },
        options: chartOptions(),
      });
    }

    const overviewRadar = document.getElementById("overviewRadarChart");
    if (overviewRadar) {
      const rtmpose = profiles.find((item) => item.model === "rtmpose") || {};
      const vitpose = profiles.find((item) => item.model === "vitpose") || {};
      const rtmposeBase = official.find((item) => item.model === "rtmpose") || {};
      const vitposeBase = official.find((item) => item.model === "vitpose") || {};
      new Chart(overviewRadar, {
        type: "radar",
        data: {
          labels: ["Speed", "Robustness", "Accuracy", "Deployability", "Stability"],
          datasets: [
            {
              label: "RTMPose",
              data: [
                Math.min(10, Number(rtmpose.avg_fps || 0) / 1.2),
                Math.min(10, 10 - Number(rtmpose.zero_people_ratio || 0) * 10),
                Number(rtmposeBase.mAP || 0) * 10,
                Math.min(10, 10 - Number(rtmpose.avg_latency_ms || 0) / 80),
                Number(rtmpose.success_ratio || 0) * 10,
              ],
              borderColor: COLORS.cyan,
              backgroundColor: "rgba(55,208,255,0.18)",
              pointBackgroundColor: COLORS.cyan,
            },
            {
              label: "ViTPose",
              data: [
                Math.min(10, Number(vitpose.avg_fps || 0) / 1.2),
                Math.min(10, 10 - Number(vitpose.zero_people_ratio || 0) * 10),
                Number(vitposeBase.mAP || 0) * 10,
                Math.min(10, 10 - Number(vitpose.avg_latency_ms || 0) / 80),
                Number(vitpose.success_ratio || 0) * 10,
              ],
              borderColor: COLORS.ember,
              backgroundColor: "rgba(255,143,90,0.16)",
              pointBackgroundColor: COLORS.ember,
            },
          ],
        },
        options: radarOptions(),
      });
    }

    const overviewPie = document.getElementById("overviewPieChart");
    if (overviewPie) {
      const usage = payload.usage_distribution || {};
      new Chart(overviewPie, {
        type: "doughnut",
        data: {
          labels: Object.keys(usage).map((key) => key.toUpperCase()),
          datasets: [
            {
              data: Object.values(usage),
              backgroundColor: [COLORS.cyan, COLORS.ember, COLORS.mint, COLORS.plum],
              borderWidth: 0,
            },
          ],
        },
        options: doughnutOptions(),
      });
    }

    const selectedVideo = getSelectedVideoRun();
    if (selectedVideo) {
      const videoFps = document.getElementById("videoFpsChart");
      if (videoFps) {
        new Chart(videoFps, {
          type: "line",
          data: {
            labels: (selectedVideo.series.fps || []).map((_, index) => index),
            datasets: [{ label: "FPS", data: selectedVideo.series.fps || [], borderColor: COLORS.cyan, backgroundColor: "rgba(55,208,255,0.15)", fill: true, tension: 0.24, pointRadius: 0 }],
          },
          options: chartOptions(),
        });
      }
      const videoLatency = document.getElementById("videoLatencyChart");
      if (videoLatency) {
        new Chart(videoLatency, {
          type: "line",
          data: {
            labels: (selectedVideo.series.latency_ms || []).map((_, index) => index),
            datasets: [{ label: "Latency (ms)", data: selectedVideo.series.latency_ms || [], borderColor: COLORS.ember, backgroundColor: "rgba(255,143,90,0.15)", fill: true, tension: 0.24, pointRadius: 0 }],
          },
          options: chartOptions(),
        });
      }
    }

    const liveQuality = document.getElementById("liveQualityChart");
    if (liveQuality) {
      const aggregate = payload.rtsp_aggregate || {};
      new Chart(liveQuality, {
        type: "doughnut",
        data: {
          labels: ["Stable", "Degraded", "Critical"],
          datasets: [
            {
              data: [aggregate.healthy_streams || 0, aggregate.degraded_streams || 0, aggregate.critical_streams || 0],
              backgroundColor: [COLORS.mint, COLORS.ember, COLORS.red],
              borderWidth: 0,
            },
          ],
        },
        options: doughnutOptions(),
      });
    }

    const severity = document.getElementById("failureSeverityChart");
    if (severity) {
      const rows = payload.failure_rows || [];
      new Chart(severity, {
        type: "pie",
        data: {
          labels: ["Stable", "Warning", "Critical"],
          datasets: [
            {
              data: [
                rows.filter((row) => row.severity === "stable").length,
                rows.filter((row) => row.severity === "warning").length,
                rows.filter((row) => row.severity === "critical").length,
              ],
              backgroundColor: [COLORS.mint, COLORS.ember, COLORS.red],
              borderWidth: 0,
            },
          ],
        },
        options: doughnutOptions(),
      });
    }

    const latencyHist = document.getElementById("failureLatencyChart");
    if (latencyHist) {
      const dist = payload.distributions || {};
      const r = buildHistogram(dist.latency?.rtmpose || [], 50);
      const v = buildHistogram(dist.latency?.vitpose || [], 50);
      const labels = r.map((row) => row.label);
      new Chart(latencyHist, {
        type: "bar",
        data: {
          labels,
          datasets: [
            { label: "RTMPose", data: r.map((row) => row.count), backgroundColor: COLORS.cyan },
            { label: "ViTPose", data: labels.map((_, index) => v[index]?.count || 0), backgroundColor: COLORS.ember },
          ],
        },
        options: chartOptions(),
      });
    }

    const confidenceHist = document.getElementById("failureConfidenceChart");
    if (confidenceHist) {
      const dist = payload.distributions || {};
      const r = buildHistogram((dist.confidence?.rtmpose || []).map((value) => value * 100), 10);
      const v = buildHistogram((dist.confidence?.vitpose || []).map((value) => value * 100), 10);
      const labels = r.map((row) => row.label);
      new Chart(confidenceHist, {
        type: "bar",
        data: {
          labels,
          datasets: [
            { label: "RTMPose", data: r.map((row) => row.count), backgroundColor: COLORS.mint },
            { label: "ViTPose", data: labels.map((_, index) => v[index]?.count || 0), backgroundColor: COLORS.plum },
          ],
        },
        options: chartOptions(),
      });
    }

    const compareLatency = document.getElementById("compareLatencyChart");
    if (compareLatency) {
      new Chart(compareLatency, {
        type: "bar",
        data: {
          labels: benchmark.map((row) => String(row.model || "").toUpperCase()),
          datasets: [
            { label: "Latency P95", data: benchmark.map((row) => Number(row.latency_ms_p95 || 0)), backgroundColor: COLORS.ember },
            { label: "Latency Mean", data: benchmark.map((row) => Number(row.latency_ms_mean || 0)), backgroundColor: COLORS.cyan },
          ],
        },
        options: chartOptions(),
      });
    }

    const compareConfidence = document.getElementById("compareConfidenceChart");
    if (compareConfidence) {
      const profiles = payload.model_profiles || [];
      new Chart(compareConfidence, {
        type: "bar",
        data: {
          labels: profiles.map((row) => String(row.model || "").toUpperCase()),
          datasets: [
            { label: "Avg Confidence", data: profiles.map((row) => Number(row.avg_confidence || 0)), backgroundColor: COLORS.plum },
            { label: "Field Success", data: profiles.map((row) => Number(row.success_ratio || 0)), backgroundColor: COLORS.mint },
          ],
        },
        options: chartOptions(),
      });
    }
  }

  function chartOptions() {
    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#cbd5e1" } },
      },
      scales: {
        x: { ticks: { color: "#8ea7c2" }, grid: { color: "rgba(255,255,255,0.05)" } },
        y: { ticks: { color: "#8ea7c2" }, grid: { color: "rgba(255,255,255,0.05)" } },
      },
    };
  }

  function radarOptions() {
    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#cbd5e1" } },
      },
      scales: {
        r: {
          angleLines: { color: "rgba(255,255,255,0.08)" },
          grid: { color: "rgba(255,255,255,0.08)" },
          pointLabels: { color: "#c8dbef" },
          ticks: { color: "#8ea7c2", backdropColor: "transparent" },
          suggestedMin: 0,
          suggestedMax: 10,
        },
      },
    };
  }

  function doughnutOptions() {
    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#cbd5e1" } },
      },
    };
  }

  renderLoading();
  fetch("/api/dashboard")
    .then((response) => {
      if (!response.ok) throw new Error(`Dashboard API failed: ${response.status}`);
      return response.json();
    })
    .then((payload) => {
      state.payload = payload;
      renderApp();
      renderCharts();
    })
    .catch((error) => renderError(error.message || "Failed to load dashboard."));
})();
