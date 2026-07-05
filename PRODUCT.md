# Product

## Register

brand

## Users

Hackathon judges and evaluators (ISRO BAH 2026, PS-4) watching a live demo, plus
disaster-response / urban-mobility analysts exploring road-network resilience.
Primary context: a projected or shared-screen demo where first impressions are scored;
secondary: hands-on interaction (stress sliders, node ablation, live location analysis).

## Product Purpose

Route Resilience turns raw satellite imagery into a routable, topologically-healed road
graph and stress-tests it (gatekeeper ablation, flood levels). The dashboard is the
showcase surface: it must make the extracted network feel like a precision instrument —
the "product on the stage" — while every number stays real (honest-metrics rule).

## Brand Personality

Cinematic · precise · mission-control. The 3D road network is the hero object under a
spotlight; instrumentation (monospace HUD readouts, measurement ticks) conveys engineering
rigor. Warm ember/gold identity on a near-black stage — confident, not neon.

## Anti-references

- Generic admin-template dashboards (Bootstrap cards, blue-on-white).
- Neon "cyberpunk hacker" dashboards — glow everywhere, unreadable.
- Sci-fi HUD cosplay: fake data, spinning radials that mean nothing. Every HUD element
  must display a real metric from the pipeline.

## Design Principles

- **The network is the product.** Layout stages the 3D graph/map like a product
  configurator stages its object; panels float around it, never bury it.
- **Instrument, don't decorate.** Monospace micro-labels, ticks, and readouts always show
  real pipeline values (RI, LCC, IoU, tile size) — never invented numbers.
- **Cinema first, tool second — but controls keep product discipline** (consistent
  affordances, visible states, 150–250 ms transitions).
- **One accent system.** Ember/gold carries actions and identity; status colors
  (good/warn/crit) are reserved for state.

## Accessibility & Inclusion

No formal WCAG target stated; keep ≥4.5:1 for body text, visible focus rings,
`prefers-reduced-motion` alternatives for all animation, keyboard operability for
interactive rows/controls (already present — preserve it).
