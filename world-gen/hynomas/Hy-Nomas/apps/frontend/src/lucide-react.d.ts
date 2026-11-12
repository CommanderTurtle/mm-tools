declare module 'lucide-react' {
  import type { ComponentType, SVGProps } from 'react'

  export type LucideIcon = ComponentType<
    SVGProps<SVGSVGElement> & { size?: number | string }
  >

  export const ArrowLeft: LucideIcon
  export const ArrowRight: LucideIcon
  export const ArrowUp: LucideIcon
  export const ArrowUpRight: LucideIcon
  export const Bot: LucideIcon
  export const Camera: LucideIcon
  export const Check: LucideIcon
  export const ChevronDown: LucideIcon
  export const Clapperboard: LucideIcon
  export const Copy: LucideIcon
  export const Download: LucideIcon
  export const Eye: LucideIcon
  export const EyeOff: LucideIcon
  export const Expand: LucideIcon
  export const Images: LucideIcon
  export const MoreHorizontal: LucideIcon
  export const Moon: LucideIcon
  export const Navigation: LucideIcon
  export const Orbit: LucideIcon
  export const Plus: LucideIcon
  export const RotateCcw: LucideIcon
  export const ScanFace: LucideIcon
  export const Share2: LucideIcon
  export const Sparkles: LucideIcon
  export const Sun: LucideIcon
  export const Upload: LucideIcon
  export const UserRound: LucideIcon
  export const X: LucideIcon
}
