/**
 * Chart color roles — validated with the dataviz six-checks script
 * (#2F5CB0 / #009999 / #EC6602 all pass on the light surface; #003087
 * failed the lightness band as a fill, so it is reserved for text/lines).
 */
export const CHART = {
  /** series 1 — observed / ground truth / baseline */
  navy: '#2F5CB0',
  /** series 2 — predicted / simulated / ours */
  teal: '#009999',
  /** highlight & mismatch only — never a plain series */
  orange: '#EC6602',
  /** de-emphasis (single-agent baselines, past phases) */
  faint: '#A9C0E4',
  grid: '#E4EFEF',
  axis: '#6B7B7B',
  ink: '#1A1A1A',
  label: '#404F4F',
} as const;

/** Body-region tints for Gantt blocks (identity carried by direct labels). */
export const REGION_FILL: Record<string, string> = {
  HEAD: '#2F5CB0',
  SPINE: '#5377C0',
  PELVIS: '#7B96CF',
  ABD: '#9FB3DD',
  EXCH: '#009999',
};

export const minutesToClock = (m: number, startHour = 8): string => {
  const h = startHour + Math.floor(m / 60);
  const mm = Math.round(m % 60);
  return `${String(h).padStart(2, '0')}:${String(mm).padStart(2, '0')}`;
};
