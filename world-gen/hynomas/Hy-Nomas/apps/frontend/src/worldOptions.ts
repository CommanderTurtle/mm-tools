import type { SelectOption, WorldStyle } from './types'

export const styleOptions: SelectOption<WorldStyle>[] = [
  {
    value: 'real-world footage',
    label: 'Real-world footage',
    description: 'Live-action footage captured in the physical world',
  },
  {
    value: 'animation/anime',
    label: 'Animation / Anime',
    description: '2D animation, anime, cartoons, or illustrated motion',
  },
  {
    value: 'CG',
    label: 'CG',
    description: 'Computer-generated imagery and 3D-rendered scenes',
  },
  {
    value: 'video game',
    label: 'Video game',
    description: 'Gameplay or footage rendered by a game engine',
  },
  {
    value: 'software/web/app/computer screen recording',
    label: 'Software / Web / App / Screen recording',
    description: 'A recording of software, a website, an app, or a computer screen',
  },
  {
    value: 'slideshow',
    label: 'Slideshow',
    description: 'A sequence of mostly static images or presentation slides',
  },
]
