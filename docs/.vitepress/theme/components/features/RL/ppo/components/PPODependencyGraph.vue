<template>
  <div class="ppo-graph-container">
    <svg class="ppo-svg" viewBox="0 0 1920 1080" preserveAspectRatio="xMidYMid meet">
      <defs>
        <!-- Gradients -->
        <linearGradient id="grad-env" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#ECFDF5" stop-opacity="0.6"/>
          <stop offset="100%" stop-color="#ECFDF5" stop-opacity="0.1"/>
        </linearGradient> 
        <filter id="glass" x="-20%" y="-20%" width="140%" height="140%">
           <feGaussianBlur in="SourceGraphic" stdDeviation="4" result="blur" />
           <feColorMatrix in="blur" type="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 18 -7" result="goo" />
        </filter>
        <filter id="soft-shadow">
          <feDropShadow dx="0" dy="4" stdDeviation="15" flood-color="rgba(0,0,0,0.05)"/>
        </filter>
        <marker id="arrow-sm" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
          <polygon points="0 0, 10 3.5, 0 7" fill="#94a3b8" />
        </marker>
      </defs>

      <!-- ===== SCENE 1 & 7: ENVIRONMENT AREA (Right Side) ===== -->
      <!-- Coords: (1200, 350) 360x360 -->
      <g transform="translate(1200, 350)">
         <rect x="0" y="0" width="360" height="360" rx="24" fill="url(#grad-env)" stroke="#A7F3D0" stroke-width="1" />
         <text x="180" y="-15" text-anchor="middle" font-size="24" fill="#059669" font-weight="600" letter-spacing="0.05em">ENVIRONMENT</text>
         
         <!-- Grid World -->
         <StateGridVisual 
            :x="180" :y="180" 
            :width="320" :height="320"
            :rows="4" :cols="4"
            :currentState="currentState"
            :nextState="nextState"
            :active="true"
         />
      </g>

      <!-- ===== SCENE 3-5: AGENT CORES (Left Split) ===== -->
      
      <!-- Actor Core (Top Left) -->
      <!-- Coords: (400, 150) -->
      <g transform="translate(400, 150)" :opacity="actorCoreOpacity" style="transition: opacity 1s">
         <rect x="-200" y="-150" width="400" height="300" rx="24" fill="#EFF6FF" stroke="#BFDBFE" stroke-opacity="0.5" />
         <text x="0" y="-120" text-anchor="middle" font-size="20" fill="#3B82F6" font-weight="bold">ACTOR $\pi_\theta$</text>
         
         <!-- Neural Net -->
         <NeuralNetworkVisual
           :x="0" :y="0"
           title="" 
           :layers="[4, 5, 5, 4]"
           color="#3B82F6"
           :activeLayer="actorActiveLayer"
           :opacity="1"
           :scale="1.2"
         />
      </g>

      <!-- Critic Core (Bottom Left) -->
      <!-- Coords: (400, 550) -->
      <g transform="translate(400, 550)" :opacity="criticCoreOpacity" style="transition: opacity 1s">
         <rect x="-200" y="-150" width="400" height="300" rx="24" fill="#F5F3FF" stroke="#DDD6FE" stroke-opacity="0.5" />
         <text x="0" y="-120" text-anchor="middle" font-size="20" fill="#8B5CF6" font-weight="bold">CRITIC $V_\phi$</text>
         
         <NeuralNetworkVisual
           :x="0" :y="0"
           title="" 
           :layers="[4, 5, 1]"
           color="#8B5CF6"
           :activeLayer="criticActiveLayer"
           :opacity="1"
           :scale="1.2"
         />
      </g>

      <!-- ===== FLOATING CONTAINERS (The "Actors") ===== -->

      <!-- 1. State Container (Main / Actor Path) -->
      <g v-if="showMainState" :transform="mainStateTransform" style="transition: transform 0.1s linear">
          <GlassContainer 
             label="s_t" 
             :values="[0.03, -0.21, 0.08, 0.00]" 
             color="#3B82F6" w="260" h="80"
             :opacity="mainStateOpacity"
          />
      </g>
      
      <!-- 2. State Container (Copy / Critic Path) -->
      <g v-if="showCopyState" :transform="copyStateTransform" style="transition: transform 0.1s linear">
          <GlassContainer 
             label="s_t (Copy)" 
             :values="[0.03, -0.21, 0.08, 0.00]" 
             color="#8B5CF6" w="260" h="80"
             :opacity="copyStateOpacity"
          />
      </g>

      <!-- 3. Action Container -->
      <g v-if="showAction" :transform="actionTransform" style="transition: transform 0.1s linear">
          <GlassContainer 
             label="a_t" 
             :values="['0', '1', '0', '0']" 
             color="#FACC15" w="200" h="70"
             :opacity="1"
             type="action"
          />
      </g>
      
      <!-- 4. Next State Container (Scene 8) -->
      <g v-if="showNextState" :transform="nextStateTransform" style="transition: transform 0.1s linear">
          <GlassContainer 
             label="s_{t+1}" 
             :values="[0.12, -0.15, 0.90, -0.01]" 
             color="#059669" w="260" h="80"
             :opacity="1"
          />
      </g>

      <!-- ===== RESULTS & MEMORY ===== -->
      
      <!-- V1 Badge -->
      <g v-if="showV1" :transform="v1Transform" style="transition: all 1s cubic-bezier(0.34, 1.56, 0.64, 1)">
         <rect x="-60" y="-20" width="120" height="40" rx="20" fill="#F3E8FF" stroke="#9333EA" stroke-width="2"/>
         <text y="6" text-anchor="middle" font-weight="bold" fill="#7E22CE">V = 12.50</text>
      </g>
      
      <!-- V2 Badge -->
      <g v-if="showV2" :transform="v2Transform" style="transition: all 1s">
         <rect x="-60" y="-20" width="120" height="40" rx="20" fill="#F3E8FF" stroke="#9333EA" stroke-width="2"/>
         <text y="6" text-anchor="middle" font-weight="bold" fill="#6B21A8">V' = 15.20</text>
      </g>
      
      <!-- Reward Pop -->
      <g v-if="showReward" transform="translate(1380, 280)" style="animation: floatUp 1s ease-out forwards">
          <text text-anchor="middle" font-size="48" font-weight="800" fill="#EAB308" filter="url(#soft-shadow)">+1.0</text>
      </g>

      <!-- Memory Dock (Bottom) -->
      <g transform="translate(500, 900)">
          <rect x="0" y="0" width="920" height="120" rx="16" fill="#F1F5F9" stroke="#E2E8F0" />
          <text x="460" y="15" text-anchor="middle" font-size="12" fill="#94A3B8" letter-spacing="0.1em">EXPERIENCE BUFFER</text>
          
          <g transform="translate(200, 60)">
              <circle r="30" fill="#E2E8F0" />
              <text y="5" text-anchor="middle" font-size="10" fill="#64748B">Empty</text>
          </g>
           <g transform="translate(300, 60)">
              <circle r="30" fill="#E2E8F0" />
          </g>
           <g transform="translate(400, 60)">
              <circle r="30" fill="#E2E8F0" />
          </g>
      </g>
      
      <!-- ===== SCENE 9: MATH BOARD ===== -->
      <g v-if="showMath" transform="translate(960, 500)">
          <rect x="-300" y="-150" width="600" height="300" rx="24" fill="rgba(255,255,255,0.9)" stroke="#FB7185" stroke-width="2" filter="url(#soft-shadow)" />
          <text y="-100" text-anchor="middle" font-size="24" fill="#334155" font-weight="bold">GAE Calculation</text>
          
          <text y="-40" text-anchor="middle" font-size="20" fill="#475569">
            δ = <tspan fill="#EAB308">1.0</tspan> + <tspan fill="#94A3B8">0.99</tspan>(<tspan fill="#8B5CF6">15.2</tspan>) - <tspan fill="#7C3AED">12.5</tspan>
          </text>
          
          <text y="20" text-anchor="middle" font-size="32" font-weight="800" fill="#FB7185">
             Advantage = 3.55
          </text>
          
          <!-- Clip Slide -->
          <g transform="translate(0, 80)">
             <line x1="-150" y1="0" x2="150" y2="0" stroke="#CBD5E1" stroke-width="4" stroke-linecap="round"/>
             <rect x="-50" y="-10" width="100" height="20" fill="#86EFAC" rx="4" opacity="0.5"/>
             <circle cx="40" cy="0" r="8" fill="#F43F5E" />
             <text y="25" text-anchor="middle" font-size="14" fill="#64748b">Ratio Clip</text>
          </g>
      </g>

    </svg>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import NeuralNetworkVisual from './NeuralNetworkVisual.vue';
import StateGridVisual from './StateGridVisual.vue';
import GlassContainer from './GlassContainer.vue';

const props = defineProps<{ animState: any }>();
const ph = computed(() => props.animState.phase);
const p = computed(() => props.animState.progress);

// --- Helpers for Transforms ---
const showMainState = computed(() => ['SCENE2_StateSpawn', 'SCENE3_Split', 'SCENE5_Actor'].includes(ph.value));
const showCopyState = computed(() => ['SCENE3_Split', 'SCENE4_Critic1'].includes(ph.value));

const mainStateTransform = computed(() => {
   if (ph.value === 'SCENE2_StateSpawn') {
       // Lift up: From Env center (1260, 490) to (1260, 430)
       const y = 490 - p.value * 60;
       return `translate(1260, ${y})`;
   }
   if (ph.value === 'SCENE3_Split') {
       // Move Left to Split Point (1100), then to Actor path
       // Simplified linear path: 1260 -> 800 (Actor approach)
       const x = 1260 - p.value * (1260 - 800);
       const y = 430 - p.value * (430 - 300);
       return `translate(${x}, ${y})`;
   }
   if (ph.value === 'SCENE5_Actor') {
       // Approach Actor Input (400, 150)
       // Let's say it starts at 800, 300 from Scene 3 end.
       const x = 800 - p.value * (800 - 450); // Stop at input
       const y = 300 - p.value * (300 - 150);
       return `translate(${x}, ${y}) scale(${1 - p.value*0.5})`; // Dissolve
   }
   // Default holding position (end of Scene 2)
   return `translate(1260, 430)`;
});

const mainStateOpacity = computed(() => ph.value === 'SCENE5_Actor' ? (1 - p.value) : 1);

const copyStateTransform = computed(() => {
   if (ph.value === 'SCENE3_Split') {
       // Split off from Main path
       // Main goes 1260 -> 800. Copy goes 1260 -> 900 (lower)
       const x = 1260 - p.value * (1260 - 900);
       const y = 430 + p.value * 70; // Drop down
       return `translate(${x}, ${y})`; 
   }
   if (ph.value === 'SCENE4_Critic1') {
       // Go to Critic Input (400, 550)
       const x = 900 - p.value * (900 - 450);
       const y = 500 + p.value * 50;
       return `translate(${x}, ${y}) scale(${1 - p.value*0.5})`;
   }
   return `translate(1260, 430)`;
});
const copyStateOpacity = computed(() => ph.value === 'SCENE4_Critic1' ? (1 - p.value) : 1);

// Action Container
const showAction = computed(() => ['SCENE6_ActionForm', 'SCENE7_Interact', 'SCENE8_Critic2'].includes(ph.value));
const actionTransform = computed(() => {
    // SCENE6: Form at Actor Output (600, 150) -> but wait, Actor is at (400,150). Output is (500, 150).
    const startX = 550; 
    const startY = 150;
    
    if (ph.value === 'SCENE6_ActionForm') return `translate(${startX}, ${startY})`;
    if (ph.value === 'SCENE7_Interact') {
        const x = startX + p.value * (1220 - startX);
        const y = startY + p.value * (450 - startY);
        return `translate(${x}, ${y})`; // Fly to grid
    }
    if (ph.value === 'SCENE8_Critic2') {
         // Fly to Buffer
         const x = 1220 - p.value * (1220 - 500);
         const y = 450 + p.value * (900 - 450);
         return `translate(${x}, ${y}) scale(0.5)`;
    }
    return `translate(${startX}, ${startY})`;
});

// Next State
const showNextState = computed(() => ['SCENE8_Critic2'].includes(ph.value));
const nextStateTransform = computed(() => {
    // Env (1300, 450) -> Critic (400, 550)
    const x = 1300 - p.value * (1300 - 450);
    const y = 450 + p.value * (550 - 450); // To Critic Input
    return `translate(${x}, ${y}) scale(${1 - p.value * 0.3})`;
});

// Results
const showV1 = computed(() => ['SCENE4_Critic1', 'SCENE5_Actor', 'SCENE6_ActionForm', 'SCENE7_Interact', 'SCENE8_Critic2', 'SCENE9_Math'].includes(ph.value));
const v1Transform = computed(() => {
    if (ph.value === 'SCENE4_Critic1') return `translate(550, 550)`; // At Critic
    // Move to Memory Dock Scene 4 end?
    return `translate(550, 920)`; // Memory position dock
});
const showV2 = computed(() => ['SCENE8_Critic2', 'SCENE9_Math'].includes(ph.value) && (ph.value !== 'SCENE8_Critic2' || p.value > 0.5));
const v2Transform = computed(() => {
    if (ph.value === 'SCENE8_Critic2') return `translate(550, 550)`; 
    return `translate(650, 920)`;
});

const showReward = computed(() => ph.value === 'SCENE7_Interact' && p.value > 0.8);
const showMath = computed(() => ph.value === 'SCENE9_Math');

// Grid States
const currentState = computed(() => ({r:1, c:1})); // Fixed for demo
const nextState = computed(() => ['SCENE7_Interact', 'SCENE8_Critic2', 'SCENE9_Math'].includes(ph.value) && (ph.value !== 'SCENE7_Interact' || p.value > 0.8) ? {r:1,c:2} : undefined);

// Neural Activations
const actorCoreOpacity = computed(() => ['SCENE1_Init', 'SCENE2_StateSpawn'].includes(ph.value) ? 0.2 : 1);
const criticCoreOpacity = computed(() => ['SCENE1_Init', 'SCENE2_StateSpawn'].includes(ph.value) ? 0.2 : 1);

const actorActiveLayer = computed(() => {
    if (ph.value === 'SCENE5_Actor') return Math.floor(p.value * 4);
    if (ph.value === 'SCENE9_Math') return 99; // Pulse
    return -1;
});
const criticActiveLayer = computed(() => {
    if (ph.value === 'SCENE4_Critic1') return Math.floor(p.value * 3);
    if (ph.value === 'SCENE8_Critic2') return Math.floor(p.value * 3);
    return -1;
});

</script>

<style scoped>
.ppo-graph-container {
  width: 100%;
  background: #F8FAFC;
  border-radius: 24px;
}
.ppo-svg {
  width: 100%;
  height: 100%;
  filter: drop-shadow(0 0 1px rgba(0,0,0,0.05));
}
</style>
