import { VueRenderer } from '@tiptap/vue-3'
import tippy from 'tippy.js'
import SlashMenu from './SlashMenu.vue'

export default {
  items: ({ query }: { query: string }) => {
    return [
      {
        title: 'Heading 1',
        description: 'Big selection heading',
        icon: 'H1',
        command: ({ editor, range }: { editor: any, range: any }) => {
          editor.chain().focus().deleteRange(range).setNode('heading', { level: 1 }).run()
        },
      },
      {
        title: 'Heading 2',
        description: 'Medium section heading',
        icon: 'H2',
        command: ({ editor, range }: { editor: any, range: any }) => {
          editor.chain().focus().deleteRange(range).setNode('heading', { level: 2 }).run()
        },
      },
      {
        title: 'Bullet List',
        description: 'Create a simple bullet list',
        icon: '•',
        command: ({ editor, range }: { editor: any, range: any }) => {
          editor.chain().focus().deleteRange(range).toggleBulletList().run()
        },
      },
      {
        title: 'Ordered List',
        description: 'Create a numbered list',
        icon: '1.',
        command: ({ editor, range }: { editor: any, range: any }) => {
          editor.chain().focus().deleteRange(range).toggleOrderedList().run()
        },
      },
      {
        title: 'Task List',
        description: 'Track tasks with a to-do list',
        icon: '☑',
        command: ({ editor, range }: { editor: any, range: any }) => {
          editor.chain().focus().deleteRange(range).toggleTaskList().run()
        },
      },
      {
        title: 'Code Block',
        description: 'Capture a code snippet',
        icon: '</>',
        command: ({ editor, range }: { editor: any, range: any }) => {
          editor.chain().focus().deleteRange(range).setCodeBlock().run()
        },
      },
      {
        title: 'Blockquote',
        description: 'Capture a quote',
        icon: '❝',
        command: ({ editor, range }: { editor: any, range: any }) => {
          editor.chain().focus().deleteRange(range).toggleBlockquote().run()
        },
      },
       {
        title: 'Formula Block',
        description: 'Insert a math formula',
        icon: '∑',
        command: ({ editor, range }: { editor: any, range: any }) => {
           // Fallback if math extension not ready
           editor.chain().focus().deleteRange(range).insertContent('$$ ').run()
        },
      },
    ].filter(item => item.title.toLowerCase().startsWith(query.toLowerCase())).slice(0, 10)
  },

  render: () => {
    let component
    let popup: any

    return {
      onStart: (props: any) => {
        component = new VueRenderer(SlashMenu, {
          props,
          editor: props.editor,
        })

        if (!props.clientRect) {
          return
        }

        popup = tippy('body', {
          getReferenceClientRect: props.clientRect,
          appendTo: () => document.body,
          content: component.element as Element,
          showOnCreate: true,
          interactive: true,
          trigger: 'manual',
          placement: 'bottom-start',
        })
      },

      onUpdate(props: any) {
        component.updateProps(props)

        if (!props.clientRect) {
          return
        }

        popup[0].setProps({
          getReferenceClientRect: props.clientRect,
        })
      },

      onKeyDown(props: any) {
        if (props.event.key === 'Escape') {
          popup[0].hide()

          return true
        }

        return component.ref?.onKeyDown(props)
      },

      onExit() {
        popup[0].destroy()
        component.destroy()
      },
    }
  },
}
