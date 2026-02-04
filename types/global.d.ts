
export {}

declare global {
  interface Window {
    loadPyodide: (config?: any) => Promise<any>
  }
}

declare module 'cytoscape-dagre' {
  const ext: any;
  export default ext;
}

