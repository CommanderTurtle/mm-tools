import { Moon, Sun } from 'lucide-react'

interface ThemeToggleProps {
  theme: 'dark' | 'light'
  onToggle: () => void
}

export function ThemeToggle({ theme, onToggle }: ThemeToggleProps) {
  const dark = theme === 'dark'

  return (
    <button
      className="theme-toggle"
      type="button"
      aria-label={`Switch to ${dark ? 'light' : 'dark'} mode`}
      title={`Switch to ${dark ? 'light' : 'dark'} mode`}
      onClick={onToggle}
    >
      {dark
        ? <Sun size={16} strokeWidth={1.8} />
        : <Moon size={16} strokeWidth={1.8} />}
    </button>
  )
}
