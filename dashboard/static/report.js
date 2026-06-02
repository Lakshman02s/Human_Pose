(function () {
  const root = document.getElementById("report-root");

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

  function section(eyebrow, title, subtitle, body) {
    return `
      <section class="glass-panel rounded-[30px] p-6 lg:p-7">
        <div class="mb-6">
          <div class="text-[11px] uppercase tracking-[0.32em] text-cyan-300/80">${eyebrow}</div>
          <div class="mt-3 text-3xl font-semibold text-white">${title}</div>
          <div class="mt-2 max-w-4xl text-sm text-slate-400">${subtitle}</div>
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
            <tr>
              ${headers.map((header) => `<th class="px-4 py-3 font-medium">${header.label}</th>`).join("")}
            </tr>
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
    root.innerHTML = `<div class="grid-shell flex min-h-screen items-center justify-center text-xl text-slate-300">Loading client report...</div>`;
  }

  function renderError(message) {
    root.innerHTML = `<div class="grid-shell flex min-h-screen items-center justify-center px-6 text-center text-red-300">${message}</div>`;
  }

  function render(payload) {
    const platform = payload.platform || {};
    const overview = platform.overview || {};
    const officialModels = (platform.official_coco_baselines && platform.official_coco_baselines.models) || [];
    const benchmarkRows = (platform.latest_benchmark || []).map((row) => ({
      model: String(row.model || "").toUpperCase(),
      fps: fmt(row.fps_mean),
      latency: `${fmt(row.latency_ms_mean)} ms`,
      cpu: `${fmt(row.cpu_percent_mean, 1)} %`,
      ram: `${fmt(row.ram_percent_mean, 1)} %`,
    }));
    const aggregateRows = (payload.aggregate_rows || []).map((row) => ({
      source: titleCase(row.source_type),
      model: String(row.model || "").toUpperCase(),
      runs: row.runs,
      fps: fmt(row.mean_fps),
      latency: `${fmt(row.mean_latency_ms)} ms`,
      people: fmt(row.mean_people),
      confidence: fmt(row.mean_keypoint_conf, 3),
      missed: fmtPct(row.mean_zero_people_ratio, 0),
    }));

    root.innerHTML = `
      <div class="grid-shell min-h-screen px-5 py-5 md:px-8 md:py-8">
        <div class="mx-auto max-w-[1650px] space-y-6">
          <div class="nav-shell flex flex-col gap-4 rounded-[30px] p-5 xl:flex-row xl:items-center xl:justify-between">
            <div>
              <div class="text-xs uppercase tracking-[0.3em] text-cyan-300/80">Client-Ready Report</div>
              <div class="mt-2 text-2xl font-semibold text-white">Pose Benchmark Review</div>
            </div>
            <div class="flex flex-wrap gap-3">
              <a href="/" class="accent-link">Back to Dashboard</a>
              <button id="printReport" class="tab-chip active">Export / Print PDF</button>
            </div>
          </div>

          <div class="glass-panel rounded-[34px] p-7">
            <div class="text-[11px] uppercase tracking-[0.32em] text-cyan-300/80">Executive Report</div>
            <h1 class="mt-4 text-5xl font-semibold leading-tight text-white">Human Pose Detection Benchmarking Report</h1>
            <p class="mt-4 max-w-4xl text-base text-slate-300">
              High-level deployment review covering official COCO accuracy baselines, recorded CCTV-style video validation, and live RTSP camera behavior.
            </p>
            <div class="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-5">
              ${statCard("Recommended Model", overview.recommended_model || "-", "Best current deployment candidate", "cyan")}
              ${statCard("Average FPS", fmt(overview.avg_fps || 0), "Cross-run throughput", "mint")}
              ${statCard("Average Latency", `${fmt(overview.avg_latency_ms || 0)} ms`, "Cross-run latency", "ember")}
              ${statCard("Live Camera", overview.live_status || "Pending", "RTSP validation status", "plum")}
              ${statCard("Total Runs", overview.total_runs || 0, "Recorded and live sessions", "cyan")}
            </div>
          </div>

          <div class="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
            ${section(
              "Executive Summary",
              "What Was Completed",
              "This section is tuned for fast client understanding without forcing them to parse raw logs.",
              `<div class="space-y-3">${(payload.executive_summary || [])
                .map((line) => `<div class="rounded-[22px] border border-white/8 bg-slate-950/25 p-4 text-sm text-slate-200">${line}</div>`)
                .join("")}</div>`
            )}
            ${section(
              "COCO Baseline",
              "Official Reference Accuracy",
              "Official benchmark numbers used to sanity-check expected 17-keypoint pose quality on the standard COCO benchmark.",
              `<div class="space-y-3">${officialModels
                .map(
                  (row) => `
                    <div class="rounded-[22px] border border-white/8 bg-slate-950/25 p-4">
                      <div class="text-lg font-semibold text-white">${String(row.model || "").toUpperCase()}</div>
                      <div class="mt-1 text-sm text-slate-400">${row.variant}</div>
                      <div class="mt-4 grid grid-cols-2 gap-3">
                        <div class="mini-metric rounded-[20px] p-3"><div class="text-xs text-slate-400">mAP</div><div class="mt-1 text-2xl font-semibold text-cyan-300">${fmt(row.mAP, 3)}</div></div>
                        <div class="mini-metric rounded-[20px] p-3"><div class="text-xs text-slate-400">mAP50</div><div class="mt-1 text-2xl font-semibold text-mint-300">${fmt(row.mAP50, 3)}</div></div>
                        <div class="mini-metric rounded-[20px] p-3"><div class="text-xs text-slate-400">mAP75</div><div class="mt-1 text-2xl font-semibold text-amber-300">${fmt(row.mAP75, 3)}</div></div>
                        <div class="mini-metric rounded-[20px] p-3"><div class="text-xs text-slate-400">Recall</div><div class="mt-1 text-2xl font-semibold text-plum-300">${fmt(row.recall, 3)}</div></div>
                      </div>
                    </div>
                  `
                )
                .join("")}</div>`
            )}
          </div>

          <div class="grid gap-6 xl:grid-cols-[1fr_1fr]">
            ${section(
              "Benchmark Snapshot",
              "Controlled RTMPose vs ViTPose Comparison",
              "A compact view of speed and resource trade-offs on the current CPU-based environment.",
              table(
                [
                  { key: "model", label: "Model" },
                  { key: "fps", label: "Mean FPS" },
                  { key: "latency", label: "Mean Latency" },
                  { key: "cpu", label: "CPU %" },
                  { key: "ram", label: "RAM %" },
                ],
                benchmarkRows
              )
            )}
            ${section(
              "Visual Summary",
              "Comparison and Validation Mix",
              "Fast visual layer for client demos and architecture discussions.",
              `
                <div class="grid gap-6 lg:grid-cols-2">
                  <div class="chart-panel rounded-[22px] p-4">
                    <div class="chart-title">Benchmark Bar Chart</div>
                    <div class="mt-4 h-[280px]"><canvas id="benchmarkChart"></canvas></div>
                  </div>
                  <div class="chart-panel rounded-[22px] p-4">
                    <div class="chart-title">Run Source Mix</div>
                    <div class="mt-4 h-[280px]"><canvas id="sourceMixChart"></canvas></div>
                  </div>
                </div>
              `
            )}
          </div>

          ${section(
            "Aggregate Evidence",
            "Combined Real-World Summary",
            "Combined metrics across recorded input videos and live RTSP validations.",
            table(
              [
                { key: "source", label: "Source" },
                { key: "model", label: "Model" },
                { key: "runs", label: "Runs" },
                { key: "fps", label: "Mean FPS" },
                { key: "latency", label: "Mean Latency" },
                { key: "people", label: "Mean People" },
                { key: "confidence", label: "Mean Confidence" },
                { key: "missed", label: "Zero-Person Ratio" },
              ],
              aggregateRows
            )
          )}

          <div class="grid gap-6 xl:grid-cols-2">
            ${section(
              "Observed Strengths",
              "Where The Pipeline Works",
              "The strongest evidence to highlight in client updates and CV review.",
              `
                <div class="space-y-3">
                  <div class="rounded-[22px] border border-white/8 bg-slate-950/25 p-4 text-sm text-slate-200">Both RTMPose and ViTPose can generate usable 17-keypoint full-body skeletons when subjects are closer, visible, and not heavily occluded.</div>
                  <div class="rounded-[22px] border border-white/8 bg-slate-950/25 p-4 text-sm text-slate-200">The pipeline runs end to end on recorded videos and on at least one live RTSP camera feed, proving practical camera-feed viability.</div>
                  <div class="rounded-[22px] border border-white/8 bg-slate-950/25 p-4 text-sm text-slate-200">RTMPose currently offers the strongest balance of speed and field usability on the available CPU environment.</div>
                </div>
              `
            )}
            ${section(
              "Observed Risks",
              "Where It Breaks",
              "The core failure modes that matter for CCTV, hospital, and surveillance deployment.",
              `<div class="space-y-3">${(payload.findings || [])
                .map((line) => `<div class="rounded-[22px] border border-white/8 bg-slate-950/25 p-4 text-sm text-slate-200">${line}</div>`)
                .join("")}</div>`
            )}
          </div>

          ${section(
            "Recommendation",
            "Deployment Decision Guidance",
            "Actionable next steps derived from the benchmark and field-validation evidence.",
            `<div class="grid gap-4 lg:grid-cols-2">${(payload.recommendations || [])
              .map((line) => `<div class="recommend-card rounded-[24px] p-5"><div class="text-sm leading-6 text-slate-200">${line}</div></div>`)
              .join("")}</div>`
          )}
        </div>
      </div>
    `;

    document.getElementById("printReport").addEventListener("click", () => window.print());

    const benchmarkChart = document.getElementById("benchmarkChart");
    if (benchmarkChart) {
      new Chart(benchmarkChart, {
        type: "bar",
        data: {
          labels: (platform.latest_benchmark || []).map((row) => String(row.model || "").toUpperCase()),
          datasets: [
            {
              label: "Mean FPS",
              data: (platform.latest_benchmark || []).map((row) => Number(row.fps_mean || 0)),
              backgroundColor: "#37d0ff",
            },
            {
              label: "Mean Latency (ms)",
              data: (platform.latest_benchmark || []).map((row) => Number(row.latency_ms_mean || 0)),
              backgroundColor: "#ff8f5a",
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { labels: { color: "#cbd5e1" } } },
          scales: {
            x: { ticks: { color: "#8ea7c2" }, grid: { color: "rgba(255,255,255,0.05)" } },
            y: { ticks: { color: "#8ea7c2" }, grid: { color: "rgba(255,255,255,0.05)" } },
          },
        },
      });
    }

    const sourceMixChart = document.getElementById("sourceMixChart");
    if (sourceMixChart) {
      new Chart(sourceMixChart, {
        type: "doughnut",
        data: {
          labels: ["Recorded", "Live RTSP"],
          datasets: [
            {
              data: [overview.recorded_runs || 0, overview.live_runs || 0],
              backgroundColor: ["#37d0ff", "#8f8bff"],
              borderWidth: 0,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { labels: { color: "#cbd5e1" } } },
        },
      });
    }
  }

  renderLoading();
  fetch("/api/report")
    .then((response) => {
      if (!response.ok) {
        throw new Error(`Report API failed: ${response.status}`);
      }
      return response.json();
    })
    .then(render)
    .catch((error) => renderError(error.message || "Failed to load report."));
})();
