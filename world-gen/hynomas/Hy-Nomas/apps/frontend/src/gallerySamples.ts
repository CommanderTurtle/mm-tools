import type { GalleryItem } from './types'

export type GallerySample = GalleryItem

export const gallerySamples: GallerySample[] = [
  {
    id: 'station-wagon',
    title: 'Blue Station Wagon',
    video: 'gallery/blue-station-wagon.mp4',
    duration: '19s',
    perspective: 'third-person follow',
    subject: 'A blue station wagon',
    attributes:
      'Side-profile view on a rural asphalt road, bordered by tall grass, autumn hedgerows, and bare deciduous trees under soft daylight.',
    direction:
      'Drive steadily along the road while the countryside streams past at a consistent speed.',
  },
  {
    id: 'alpine-convoy',
    title: 'Alpine Armored Convoy',
    video: 'gallery/alpine-armored-convoy.mp4',
    duration: '38s',
    perspective: 'third-person follow',
    subject: 'A military armored truck',
    attributes:
      'Heavy olive vehicle on a winding gravel track through a dramatic alpine valley with rocky slopes, sparse pines, smoke, and distant snow peaks.',
    direction:
      'Round the mountain bend, kick up dust, and continue toward the dark tunnel entrance.',
  },
  {
    id: 'canal-walk',
    title: 'Canal-Side Walk',
    video: 'gallery/canal-side-walk.mp4',
    duration: '66s',
    perspective: 'first-person',
    subject: 'A continuous first-person journey beside a quiet canal',
    attributes:
      'A narrow dirt path between tall grass, an industrial brick building with graffiti, calm reflective water, and a white pipeline gantry ahead.',
    direction:
      'Move forward naturally along the path as the canal, buildings, and vegetation evolve continuously.',
  },
  {
    id: 'coastal-flight',
    title: 'Coastal City Flight',
    video: 'gallery/coastal-city-flight.mp4',
    duration: '52s',
    perspective: 'third-person follow',
    subject: 'An ANA-liveried wide-body passenger jet',
    attributes:
      'The aircraft flies above a sprawling simulated coastal metropolis, river, highways, warehouses, and dense city blocks in crisp daylight.',
    direction:
      'Maintain steady level flight while the aircraft shadow moves across the city below.',
  },
  {
    id: 'desert-drive',
    title: 'Desert City Drive',
    video: 'gallery/desert-city-drive.mp4',
    duration: '62s',
    perspective: 'first-person',
    subject: 'A first-person drive through a sunlit desert town',
    attributes:
      'Cracked asphalt, traffic lights, palm-lined medians, low storefronts, a roadside gas station, and distant mountains under a clear blue sky.',
    direction:
      'Continue through the intersection and follow the boulevard as the roadside environment changes.',
  },
  {
    id: 'river-raft',
    title: 'Desert River Raft',
    video: 'gallery/desert-river-raft.mp4',
    duration: '19s',
    perspective: 'third-person follow',
    subject: 'A river rafter in a loaded blue raft',
    attributes:
      'The raft travels through a wide desert river surrounded by red sandstone mesas, sparse shrubs, green shoreline trees, and bright reflected sunlight.',
    direction:
      'Pull both oars and guide the raft downstream through the churning brown-water riffles.',
  },
]
