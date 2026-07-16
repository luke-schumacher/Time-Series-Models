import type { ActDef } from './types';

/** Talk structure — time budgets sum to 45 min (+15 min Q&A, appendix on demand). */
export const ACTS: ActDef[] = [
  { id: 0, label: 'Opening', short: 'Open', minutes: 3 },
  { id: 1, label: 'The Mandate & the Problem', short: 'Problem', minutes: 5 },
  { id: 2, label: 'What We Built', short: 'Built', minutes: 6 },
  { id: 3, label: 'How It Works', short: 'How', minutes: 7 },
  { id: 4, label: 'Demonstrated Results', short: 'Results', minutes: 9 },
  { id: 5, label: 'Thesis · Agentic Infra Co-Pilot', short: 'Thesis', minutes: 5 },
  { id: 6, label: 'Operations & Cloud', short: 'Cloud', minutes: 4 },
  { id: 7, label: 'Roadmap & the Ask', short: 'Ask', minutes: 6 },
  { id: 8, label: 'Appendix — on demand', short: 'Appendix', minutes: 0 },
];
