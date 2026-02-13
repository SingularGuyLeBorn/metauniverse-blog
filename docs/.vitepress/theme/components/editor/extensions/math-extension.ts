import { Node, mergeAttributes } from '@tiptap/core'
import { VueNodeViewRenderer } from '@tiptap/vue-3'
import MathView from './MathView.vue'
import { InputRule } from '@tiptap/core'

export const MathExtension = Node.create({
  name: 'math',

  group: 'inline',
  
  inline: true,
  
  selectable: true,

  atom: true,

  addAttributes() {
    return {
      src: {
        default: '',
      },
      display: {
        default: 'inline', // 'inline' or 'block'
      },
    }
  },

  parseHTML() {
    return [
      {
        tag: 'math-component',
      },
    ]
  },

  renderHTML({ HTMLAttributes }) {
    return ['math-component', mergeAttributes(HTMLAttributes)]
  },

  addNodeView() {
    return VueNodeViewRenderer(MathView)
  },

  addInputRules() {
    return [
      // Inline math: $...$
      new InputRule({
        find: /\$(.+)\$$/,
        handler: ({ state, range, match }) => {
          const { tr } = state
          const start = range.from
          const end = range.to
          const src = match[1]

          tr.replaceWith(start, end, this.type.create({ src, display: 'inline' }))
        },
      }),
      // Block math: $$ (trigger)
      new InputRule({
        find: /^\$\$\s$/,
        handler: ({ state, range, match }) => {
            const { tr } = state
            const start = range.from
            const end = range.to
            
            tr.replaceWith(start, end, this.type.create({ src: '', display: 'block' }))
        }
      })
    ]
  },
})
