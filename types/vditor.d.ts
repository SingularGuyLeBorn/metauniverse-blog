declare module 'vditor' {
  export default class Vditor {
    constructor(id: string | HTMLElement, options?: any)
    public setValue(value: string): void
    public getValue(): string
    public insertValue(value: string): void
    public focus(): void
    public blur(): void
    public disabled(): void
    public enable(): void
    public setTheme(theme: string, contentTheme: string, codeTheme: string, contentStyle?: string): void
    public destroy(): void
    public toPreview(value: string, element: HTMLElement, options?: any): void
  }
}
