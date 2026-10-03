---
name: London Network Planner
description: A dense, evidence-led GIS workspace for designing and operating hypothetical metro networks.
colors:
  action-blue: "#0b66d4"
  action-blue-deep: "#084d9f"
  action-blue-soft: "#e8f1fc"
  planning-ink: "#17202b"
  secondary-ink: "#52606d"
  muted-ink: "#5e6a75"
  paper: "#ffffff"
  work-surface: "#f4f6f7"
  rule: "#d6dde1"
  evidence-amber: "#b86608"
  unsupported-red: "#b42332"
  coverage-green: "#14836f"
typography:
  metric:
    fontFamily: "Aptos, Segoe UI, system-ui, sans-serif"
    fontSize: "34px"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "-0.035em"
  title:
    fontFamily: "Aptos, Segoe UI, system-ui, sans-serif"
    fontSize: "17px"
    fontWeight: 700
    lineHeight: 1.2
  body:
    fontFamily: "Aptos, Segoe UI, system-ui, sans-serif"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.4
  label:
    fontFamily: "Aptos, Segoe UI, system-ui, sans-serif"
    fontSize: "11px"
    fontWeight: 700
    lineHeight: 1.4
rounded:
  control: "5px"
  surface: "8px"
  dialog: "10px"
  pill: "999px"
spacing:
  compact: "6px"
  control: "9px"
  standard: "12px"
  section: "16px"
components:
  button-primary:
    backgroundColor: "{colors.action-blue}"
    textColor: "{colors.paper}"
    rounded: "{rounded.control}"
    padding: "0 12px"
    height: "34px"
  button-secondary:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.planning-ink}"
    rounded: "{rounded.control}"
    padding: "0 12px"
    height: "34px"
  input:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.planning-ink}"
    rounded: "{rounded.control}"
    padding: "0 9px"
    height: "34px"
  status-chip:
    backgroundColor: "{colors.work-surface}"
    textColor: "{colors.secondary-ink}"
    rounded: "{rounded.pill}"
    padding: "0 7px"
    height: "22px"
---

# Design System: London Network Planner

## Overview

**Creative North Star: "The Maintained Planning Desk"**

The interface behaves like a city planner's working GIS document: compact, calm, information-dense, and visibly maintained. Paper-white work surfaces frame a continuous geographic canvas, while rules, tabular figures, infrastructure line patterns, and one safety-blue action colour make the model inspectable without making it theatrical.

Depth is shallow and operational. Strong colour is semantic: blue acts, green confirms evidence coverage, amber marks incomplete evidence, and red marks unsupported or failed states. The map remains the focal object at every desktop size.

**Key Characteristics:**

- Dense three-pane desktop geography with a persistent analysis drawer
- Compact controls, crisp one-pixel rules, and tabular operational figures
- Pattern and text reinforce every colour-coded planning state
- Restrained motion reserved for state changes and live train movement

## Colors

The palette combines cool paper neutrals with sparse, semantic transport colours.

### Primary

- **Safety Action Blue:** Drives primary actions, selected tabs, focus, and active tools.
- **Deep Action Blue:** Carries hover and high-contrast selected text.
- **Pale Selection Blue:** Marks selected rows and active segmented controls without competing with the map.

### Secondary

- **Evidence Amber:** Identifies neutral PTAL fallback and incomplete evidence.
- **Unsupported Red:** Identifies invalid geography, failures, closures, and critical issues.
- **Coverage Green:** Defines supported planning boundaries and successful states.

### Neutral

- **Planning Ink:** Primary text and structural marks.
- **Secondary Ink:** Supporting copy and inactive controls.
- **Muted Ink:** Metadata that must remain readable at compact sizes.
- **Paper and Work Surface:** Alternate foreground panels and the application field.
- **Rule Grey:** Dividers, field strokes, and table structure.

### Named Rules

**The Semantic Colour Rule.** Blue acts, green validates, amber qualifies, and red blocks; do not reuse those colours decoratively.

**The Map Legibility Rule.** Coverage overlays remain translucent, bounded, and paired with a pattern or line treatment so geography stays readable beneath them.

## Typography

**Display Font:** Aptos (with Segoe UI and system sans-serif fallbacks)
**Body Font:** Aptos (with Segoe UI and system sans-serif fallbacks)
**Label/Mono Font:** SFMono Regular only where machine-readable data requires it

**Character:** The type system is restrained and administrative rather than promotional. Weight, spacing, and tabular numerals establish hierarchy within a deliberately compact scale.

### Hierarchy

- **Metric** (700, 34px, 1): Network score and journey totals only.
- **Title** (700, 17px, 1.2): Dialog and high-level task titles.
- **Body** (400, 13px, 1.4): Instructions and planning explanations.
- **Label** (700, 11px, 1.4): Field labels, metadata, table headings, and compact state labels.

### Named Rules

**The Operational Type Rule.** Use size for primary results, weight for control hierarchy, and tabular figures for values; avoid display typography that turns the workspace into a presentation page.

## Layout

The workspace uses a fixed 52px project bar over a three-column desktop grid: a 316px workflow panel, a flexible map with a 420px minimum, and a 336px inspector. At narrower desktop widths the side panels reduce to 292px and 310px. The analysis surface is a 278px drawer inset 12px from the map edges.

Spacing follows a compact rhythm built around 6px, 9px, 12px, and 16px. Related controls remain tight; section boundaries and one-pixel rules provide separation. Below 1100px, the planning surface is replaced by a clear desktop-workspace requirement rather than degrading the map and evidence into unusable stacked panels.

## Elevation & Depth

The system is flat by default. Tonal layering and rules separate permanent panels; soft, downward shadows are reserved for floating map controls, popups, the analysis drawer, and dialogs.

### Shadow Vocabulary

- **Floating Control** (`0 2px 8px rgba(23, 32, 43, 0.12)`): Map tools, notices, and keys.
- **Workspace Overlay** (`0 5px 18px rgba(23, 32, 43, 0.12)`): Analysis drawer and the narrow-screen gate.
- **Protected Focus** (`0 18px 60px rgba(23, 32, 43, 0.25)`): Modal method documentation only.

### Named Rules

**The Flat-By-Default Rule.** Permanent work surfaces use either a rule or tonal contrast; shadows indicate a surface floating over the map.

## Shapes

Corners are compact and functional: 5px for controls, 8px for workspace surfaces, and 10px for dialogs. Pills are reserved for short status chips. Stations remain circular unless closed, when a rotated square changes both silhouette and colour. Infrastructure uses solid underground strokes and dashed overground strokes.

## Components

### Buttons

- **Shape:** Compact rectangular controls with gently curved corners (5px) and a 34px minimum height.
- **Primary:** Safety blue with white text and 12px horizontal padding.
- **Hover / Focus:** Deepen the action colour on hover; use a two-pixel blue focus outline with a two-pixel offset.
- **Secondary:** White or transparent surfaces with a one-pixel structural stroke.

### Chips

- **Style:** Small textual states with a 22px minimum height and full pill radius.
- **State:** Neutral, success, warning, and danger variants always carry explicit words, never colour alone.

### Cards / Containers

- **Corner Style:** 6–8px for list rows and floating work surfaces.
- **Background:** Paper over the cool work surface.
- **Shadow Strategy:** Only map-floating containers receive elevation.
- **Border:** One-pixel rules on flat rows; elevated surfaces rely on shadow.
- **Internal Padding:** 9–16px according to information density.

### Inputs / Fields

- **Style:** White field, one-pixel strong-grey stroke, 5px radius, and 34px height.
- **Focus:** Global two-pixel safety-blue outline.
- **Error / Disabled:** Error copy uses unsupported red; disabled controls reduce opacity and retain their label.

### Navigation

The project bar is deep slate with compact outlined actions. Design/Operate and analysis tabs use pale selection blue, deep-blue text, and an inset blue rule. The narrow-screen state replaces navigation with a single explanatory gate.

### Map Evidence

Supported London uses a green dashed boundary. Unsupported areas use a translucent red mask and matching hatched legend; neutral PTAL fallback uses amber hatching. Underground and overground segments differ by solid and dashed strokes as well as labels.

## Do's and Don'ts

### Do:

- **Do** preserve the map as the dominant continuous work surface.
- **Do** pair colour with labels, patterns, line styles, or shapes.
- **Do** keep model assumptions and simulated-data labels beside their results.
- **Do** use tabular numerals for scores, times, frequencies, and demand.

### Don't:

- **Don't** turn metrics into a dashboard grid of detached cards.
- **Don't** use red, amber, green, or blue decoratively outside their semantic roles.
- **Don't** stack the full planning workspace on narrow screens; use the explicit desktop gate.
- **Don't** introduce ornamental illustration, gradients, or consumer-map styling into the maintained GIS world.
