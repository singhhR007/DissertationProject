import PageHeader from "../components/ui/PageHeader";
import Badge from "../components/ui/Badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../components/ui/Card";

type MetricRow = {
  model: string;
  threshold: string;
  precision: string;
  recall: string;
  f1: string;
  prAuc: string;
  rocAuc: string;
  brier: string;
};

const fullHdfsRows: MetricRow[] = [
  {
    model: "Baseline",
    threshold: "0.9235",
    precision: "0.9583",
    recall: "0.9950",
    f1: "0.9763",
    prAuc: "0.9800",
    rocAuc: "0.9996",
    brier: "0.0023",
  },
  {
    model: "Calibrated",
    threshold: "0.3877",
    precision: "0.9013",
    recall: "1.0000",
    f1: "0.9481",
    prAuc: "0.9764",
    rocAuc: "0.9995",
    brier: "0.0027",
  },
  {
    model: "Enriched",
    threshold: "0.9622",
    precision: "0.9444",
    recall: "0.9638",
    f1: "0.9540",
    prAuc: "0.9702",
    rocAuc: "0.9994",
    brier: "0.0030",
  },
];

const smokeTestRows: MetricRow[] = [
  {
    model: "TF-IDF + LR (Baseline)",
    threshold: "0.7856",
    precision: "0.9182",
    recall: "0.9966",
    f1: "0.9558",
    prAuc: "0.9617",
    rocAuc: "0.9993",
    brier: "0.0027",
  },
  {
    model: "BiLSTM",
    threshold: "0.8324",
    precision: "0.8797",
    recall: "0.9488",
    f1: "0.9130",
    prAuc: "0.9475",
    rocAuc: "0.9966",
    brier: "0.0136",
  },
];

function MetricTable({ rows }: { rows: MetricRow[] }) {
  return (
    <div className="overflow-x-auto rounded-2xl border border-slate-200">
      <table className="min-w-full divide-y divide-slate-200 text-sm">
        <thead className="bg-slate-50 text-left text-slate-600">
          <tr>
            <th className="px-4 py-3 font-medium">Model</th>
            <th className="px-4 py-3 font-medium">Threshold</th>
            <th className="px-4 py-3 font-medium">Precision</th>
            <th className="px-4 py-3 font-medium">Recall</th>
            <th className="px-4 py-3 font-medium">F1</th>
            <th className="px-4 py-3 font-medium">PR-AUC</th>
            <th className="px-4 py-3 font-medium">ROC-AUC</th>
            <th className="px-4 py-3 font-medium">Brier</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-200 bg-white text-slate-700">
          {rows.map((row) => (
            <tr key={row.model}>
              <td className="px-4 py-3 font-medium text-slate-900">{row.model}</td>
              <td className="px-4 py-3">{row.threshold}</td>
              <td className="px-4 py-3">{row.precision}</td>
              <td className="px-4 py-3">{row.recall}</td>
              <td className="px-4 py-3">{row.f1}</td>
              <td className="px-4 py-3">{row.prAuc}</td>
              <td className="px-4 py-3">{row.rocAuc}</td>
              <td className="px-4 py-3">{row.brier}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ConfusionMatrixCard({
  title,
  subtitle,
  tn,
  fp,
  fn,
  tp,
}: {
  title: string;
  subtitle: string;
  tn: number;
  fp: number;
  fn: number;
  tp: number;
}) {
  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{subtitle}</CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 overflow-hidden rounded-2xl border border-slate-200 text-center text-sm">
          <div className="border-b border-r border-slate-200 bg-slate-50 p-3 font-medium text-slate-600">
            True normal → Pred normal
          </div>
          <div className="border-b border-slate-200 bg-slate-50 p-3 font-medium text-slate-600">
            True normal → Pred anomalous
          </div>
          <div className="border-r border-slate-200 p-5 text-2xl font-semibold text-slate-900">
            {tn}
          </div>
          <div className="p-5 text-2xl font-semibold text-slate-900">{fp}</div>

          <div className="border-y border-r border-slate-200 bg-slate-50 p-3 font-medium text-slate-600">
            True anomalous → Pred normal
          </div>
          <div className="border-y border-slate-200 bg-slate-50 p-3 font-medium text-slate-600">
            True anomalous → Pred anomalous
          </div>
          <div className="border-r border-slate-200 p-5 text-2xl font-semibold text-slate-900">
            {fn}
          </div>
          <div className="p-5 text-2xl font-semibold text-slate-900">{tp}</div>
        </div>

        <p className="text-sm text-slate-600">
          TN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}
        </p>
      </CardContent>
    </Card>
  );
}

export default function EvaluationPage() {
  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="Research results"
        title="Evaluation"
        description="This page summarises the results reported in Chapter 5, covering the full HDFS classical model evaluation, threshold behaviour, deployment rationale and the exploratory 50k smoke test against a BiLSTM sequence model."
        actions={<Badge variant="info">Chapter 5 results</Badge>}
      />

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Card>
          <CardHeader>
            <CardTitle>Best full-HDFS model</CardTitle>
            <CardDescription>Classical comparison</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-semibold text-slate-950">Baseline</p>
            <p className="mt-2 text-sm text-slate-600">
              Message-based TF-IDF + Logistic Regression ranked strongest overall.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Best full-HDFS F1</CardTitle>
            <CardDescription>Tuned threshold</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-semibold text-slate-950">0.9763</p>
            <p className="mt-2 text-sm text-slate-600">
              Achieved by the baseline at threshold 0.9235.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Deployment model</CardTitle>
            <CardDescription>Engineering decision</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-semibold text-slate-950">Calibrated</p>
            <p className="mt-2 text-sm text-slate-600">
              Selected for the API because the risk_score field was intended to
              behave like a probability-oriented indicator.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>50k smoke-test winner</CardTitle>
            <CardDescription>Baseline vs BiLSTM</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-semibold text-slate-950">Baseline</p>
            <p className="mt-2 text-sm text-slate-600">
              The classical baseline outperformed the BiLSTM across every reported metric.
            </p>
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Full HDFS results for the classical models</CardTitle>
            <CardDescription>
              Tuned threshold comparison of the baseline, calibrated and enriched
              configurations on the full HDFS test set.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <MetricTable rows={fullHdfsRows} />
            <div className="grid gap-4 lg:grid-cols-3">
              <div className="rounded-2xl border border-slate-200 p-4">
                <p className="text-sm font-medium text-slate-900">Baseline</p>
                <p className="mt-2 text-sm leading-6 text-slate-600">
                  Strongest overall on F1, PR-AUC, ROC-AUC and Brier score.
                </p>
              </div>
              <div className="rounded-2xl border border-slate-200 p-4">
                <p className="text-sm font-medium text-slate-900">Calibrated</p>
                <p className="mt-2 text-sm leading-6 text-slate-600">
                  Reached perfect recall but underperformed the baseline on precision,
                  F1 and Brier score.
                </p>
              </div>
              <div className="rounded-2xl border border-slate-200 p-4">
                <p className="text-sm font-medium text-slate-900">Enriched</p>
                <p className="mt-2 text-sm leading-6 text-slate-600">
                  Additional structured tokens did not improve the HDFS result and
                  appear to have weakened the feature representation.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
        <Card>
          <CardHeader>
            <CardTitle>Threshold behaviour and operational decision quality</CardTitle>
            <CardDescription>
              Baseline threshold tuning produced the largest single improvement in the study.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 text-sm leading-7 text-slate-600">
            <p>
              At the default threshold of 0.5, the baseline achieved precision 0.9020,
              recall 0.9997, and F1-score 0.9483. After tuning on the validation set,
              performance improved to precision 0.9583, recall 0.9950 and F1-score 0.9763.
            </p>
            <p>
              This tuning reduced false positives from 366 to 146 while increasing
              false negatives from 1 to 17. That corresponds to roughly a 60% reduction
              in false alarms at the cost of 16 additional missed anomalies.
            </p>
            <p>
              In an API setting, this trade-off was considered worthwhile because
              unnecessary alerts reduce operator trust and create downstream review effort.
            </p>
            <div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-4">
              <p className="text-sm font-medium text-emerald-900">
                Selected operating point
              </p>
              <p className="mt-2 text-sm leading-6 text-emerald-800">
                The baseline F1 optimum was approximately 0.94 at a threshold of about 0.93,
                which aligns closely with the chosen tuned threshold of 0.9235.
              </p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Probability quality and deployment rationale</CardTitle>
            <CardDescription>
              Deployment criteria were not identical to offline classifier ranking.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 text-sm leading-7 text-slate-600">
            <p>
              The full-HDFS Brier scores were 0.0023 for the baseline, 0.0027 for the
              calibrated model and 0.0030 for the enriched model.
            </p>
            <p>
              Even though the calibrated variant did not outperform the baseline on Brier
              score, it was still selected for API deployment because the exposed
              <code className="mx-1 rounded bg-slate-100 px-1 py-0.5 text-xs">risk_score</code>
              field was intended to be interpreted as a probability like value rather
              than a raw classifier score.
            </p>
            <p>
              This was documented as an engineering judgement rather than a purely
              metric driven conclusion.
            </p>
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-6 md:grid-cols-2">
        <ConfusionMatrixCard
          title="Baseline confusion matrix — default threshold"
          subtitle="Full held-out HDFS test set at threshold 0.5"
          tn={111279}
          fp={366}
          fn={1}
          tp={3367}
        />
        <ConfusionMatrixCard
          title="Baseline confusion matrix — tuned threshold"
          subtitle="Full held-out HDFS test set at threshold 0.9235"
          tn={111499}
          fp={146}
          fn={17}
          tp={3351}
        />
      </section>

      <section className="grid gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Exploratory 50k smoke test: baseline vs BiLSTM</CardTitle>
            <CardDescription>
              Comparative study on the held-out HDFS 50k smoke-test subset.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <MetricTable rows={smokeTestRows} />
            <div className="rounded-2xl border border-slate-200 p-4 text-sm leading-7 text-slate-600">
              <p>
                The tuned baseline_50k achieved precision 0.9182, recall 0.9966,
                F1-score 0.9558, PR-AUC 0.9617, ROC-AUC 0.9993, and Brier score 0.0027.
              </p>
              <p className="mt-3">
                The tuned BiLSTM_50k achieved precision 0.8797, recall 0.9488,
                F1-score 0.9130, PR-AUC 0.9475, ROC-AUC 0.9966, and Brier score 0.0136.
              </p>
              <p className="mt-3">
                The classical baseline outperformed the BiLSTM across every reported
                metric, and the Brier gap was particularly large.
              </p>
            </div>
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-6 md:grid-cols-2">
        <ConfusionMatrixCard
          title="50k smoke test — TF-IDF baseline"
          subtitle="Held-out HDFS 50k subset"
          tn={9681}
          fp={26}
          fn={1}
          tp={292}
        />
        <ConfusionMatrixCard
          title="50k smoke test — BiLSTM"
          subtitle="Held-out HDFS 50k subset"
          tn={9669}
          fp={38}
          fn={15}
          tp={278}
        />
      </section>

      <section className="grid gap-6 xl:grid-cols-[1fr_1fr]">
        <Card>
          <CardHeader>
            <CardTitle>Interpretation</CardTitle>
            <CardDescription>
              Summary of what the results imply for the artefact.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 text-sm leading-7 text-slate-600">
            <p>
              The message based TF-IDF + Logistic Regression baseline remained the
              strongest model family in the project, outperforming both the calibrated
              and enriched variants in the main full-HDFS study.
            </p>
            <p>
              Threshold tuning proved operationally important because it changed the
              false-positive burden substantially without overturning the underlying
              model ranking.
            </p>
            <p>
              The exploratory deep learning follow up did not show enough empirical
              benefit to justify a full scale BiLSTM training run.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Consolidated findings</CardTitle>
            <CardDescription>
              Main takeaways documented in Chapter 5.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ol className="space-y-3 text-sm text-slate-600">
              <li className="rounded-xl border border-slate-200 px-4 py-3">
                1. The message-based TF-IDF + Logistic Regression baseline was the
                strongest classical model on the full HDFS dataset.
              </li>
              <li className="rounded-xl border border-slate-200 px-4 py-3">
                2. Validation set threshold tuning produced the largest single
                operational improvement in the study.
              </li>
              <li className="rounded-xl border border-slate-200 px-4 py-3">
                3. The calibrated model was deployed to the API for interpretability
                reasons, even though it did not outperform the baseline offline.
              </li>
              <li className="rounded-xl border border-slate-200 px-4 py-3">
                4. The 50k smoke test did not justify extending the deep learning
                study to a full BiLSTM training run.
              </li>
            </ol>
          </CardContent>
        </Card>
      </section>
    </div>
  );
}