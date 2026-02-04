<template>
  <div class="not-found">
    <div class="content">
      <div class="glitch" data-text="404">404</div>
      <h1>Lost in MetaUniverse</h1>
      <p>看来你来到了知识荒原。</p>
      
      <div class="actions">
        <a href="/" class="home-btn">返回母舰</a>
      </div>
    </div>
    
    <div class="stars"></div>
  </div>
</template>

<style scoped>
.not-found {
  min-height: 80vh;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  position: relative;
  overflow: hidden;
  background: radial-gradient(circle at center, #1b2735 0%, #090a0f 100%);
  color: #fff;
}

.content {
  z-index: 2;
  position: relative;
}

.glitch {
  font-size: 8rem;
  font-weight: bold;
  position: relative;
  text-shadow: 0.05em 0 0 #00fffc, -0.03em -0.04em 0 #fc00ff,
    0.025em 0.04em 0 #fffc00;
  animation: glitch 725ms infinite;
}

.glitch::before,
.glitch::after {
  content: attr(data-text);
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
}

h1 {
  font-size: 2rem;
  margin: 1rem 0;
  color: #e0e0e0;
}

p {
  color: #a0a0a0;
  margin-bottom: 2rem;
}

.home-btn {
  display: inline-block;
  padding: 0.8rem 2rem;
  background: var(--vp-c-brand-1);
  color: white;
  border-radius: 999px;
  text-decoration: none;
  font-weight: 600;
  transition: transform 0.2s, box-shadow 0.2s;
}

.home-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 0 20px rgba(var(--vp-c-brand-1-rgb), 0.5);
}

@keyframes glitch {
  0% { text-shadow: 0.05em 0 0 #00fffc, -0.03em -0.04em 0 #fc00ff, 0.025em 0.04em 0 #fffc00; }
  15% { text-shadow: 0.05em 0 0 #00fffc, -0.03em -0.04em 0 #fc00ff, 0.025em 0.04em 0 #fffc00; }
  16% { text-shadow: -0.05em -0.025em 0 #00fffc, 0.025em 0.035em 0 #fc00ff, -0.05em -0.05em 0 #fffc00; }
  49% { text-shadow: -0.05em -0.025em 0 #00fffc, 0.025em 0.035em 0 #fc00ff, -0.05em -0.05em 0 #fffc00; }
  50% { text-shadow: 0.05em 0.035em 0 #00fffc, 0.03em 0 0 #fc00ff, 0 -0.04em 0 #fffc00; }
  99% { text-shadow: 0.05em 0.035em 0 #00fffc, 0.03em 0 0 #fc00ff, 0 -0.04em 0 #fffc00; }
  100% { text-shadow: -0.05em 0 0 #00fffc, -0.025em -0.04em 0 #fc00ff, -0.04em -0.025em 0 #fffc00; }
}
</style>
