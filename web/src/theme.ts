import theme from "../../configs/visual-theme.json";

export const agentColors: Record<string, string> = theme.agents;
export default theme;

// Historical recordings retain the original default colors. Re-skin those
// defaults while preserving an explicitly customized agent color.
const legacyDefaults = new Set(["#9ef07e", "#a99aff", "#ffc878"]);
export function agentColor(id: string, color?: string): string {
  if (color && !legacyDefaults.has(color.toLowerCase())) return color;
  return agentColors[id] ?? theme.sage;
}
