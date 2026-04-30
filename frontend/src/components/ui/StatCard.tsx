import { Card, CardContent, CardHeader, CardTitle } from "./Card";
import Badge from "./Badge";

type StatCardProps = {
  label: string;
  value: string;
  helper?: string;
  trend?: string;
  trendVariant?: "neutral" | "success" | "warning" | "danger" | "info";
};

export default function StatCard({
  label,
  value,
  helper,
  trend,
  trendVariant = "neutral",
}: StatCardProps) {
  return (
    <Card className="h-full">
      <CardHeader className="flex flex-row items-start justify-between gap-4">
        <div>
          <p className="text-sm font-medium text-slate-500">{label}</p>
          <CardTitle className="mt-3 text-3xl font-semibold">{value}</CardTitle>
        </div>

        {trend ? <Badge variant={trendVariant}>{trend}</Badge> : null}
      </CardHeader>

      {helper ? (
        <CardContent>
          <p className="text-sm text-slate-500">{helper}</p>
        </CardContent>
      ) : null}
    </Card>
  );
}