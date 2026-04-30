import { NavLink } from "react-router";
import { NAV_ITEMS } from "../../lib/navigation";
import { PROJECT_META } from "../../lib/project";
import { cn } from "../../lib/utils";

type SidebarProps = {
  mobile?: boolean;
  onNavigate?: () => void;
};

export default function Sidebar({
  mobile = false,
  onNavigate,
}: SidebarProps) {
  return (
    <aside
      className={cn(
        "flex flex-col bg-white",
        mobile
          ? "h-full w-[18rem] border-r border-slate-200"
          : "sticky top-0 hidden h-screen w-72 shrink-0 border-r border-slate-200 lg:flex"
      )}
    >
      <div className="border-b border-slate-200 px-6 py-6">
        <div className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-slate-900 text-sm font-semibold text-white">
            LAD
          </div>

          <div className="min-w-0">
            <p className="truncate text-sm font-semibold text-slate-900">
              {PROJECT_META.shortName}
            </p>
            <p className="text-xs text-slate-500">{PROJECT_META.subtitle}</p>
          </div>
        </div>
      </div>

      <nav className="flex-1 space-y-1 px-4 py-6">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            className="block"
            onClick={onNavigate}
          >
            {({ isActive }) => (
              <span
                className={cn(
                  "flex items-center rounded-xl px-4 py-3 text-sm font-medium transition",
                  isActive
                    ? "bg-slate-900 !text-white shadow-sm"
                    : "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
                )}
              >
                {item.label}
              </span>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="mt-auto border-t border-slate-200 p-4">
        <div className="rounded-2xl bg-slate-50 p-4">
          <p className="text-sm font-semibold text-slate-900">
            {PROJECT_META.environmentLabel}
          </p>
          <p className="mt-1 text-sm leading-6 text-slate-600">
            The frontend and API are currently intended for local demonstration,
            integration and evaluation.
          </p>
        </div>
      </div>
    </aside>
  );
}