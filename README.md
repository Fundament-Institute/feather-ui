# Feather GUI
Feather is a lightweight GUI abstraction layer that separates functionality from implementation. By internalizing all GUI behavior, implementations of Feather need only to render widget components properly and handle any OS-specific needs. This allows Feather layouts to be opened as both native executables or embedded in games without needing any special treatment. Written in minimal C, Feather is simple to integrate into higher-level languages.

## A Unification of UI Concepts

FeatherGUI is the next step in the evolution of modern UI design. It combines the lightweight flexibility of Nuklear with the native tooling support of Qt and the ease of use of HTML. FeatherGUI is not just another widget toolkit, it serves as an abstraction layer for the entire user interface, a layer *above* other UI libraries. By seperating the underlying UI logic from the layout and behavior, FeatherGUI delivers the first **truly language agnostic UI standard**.

Like Nuklear, you can write your own backend for FeatherGUI, making it easy to integrate into your game engine. However, like Qt, a backend for FeatherGUI can be implemented using **native operating system controls**, and can piggy-back on the operating system's own message pump. This is all controlled by a **standard binary layout format**, which can be serialized and parsed to the file format of your choice - an XML schema is already included. 

All this, combined with FeatherGUI's native C interface means that you no longer have to wait for your favorite language or operating system to be supported. FeatherGUI is completely platform-independent and language-agnostic, allowing you to implement a binding for your favorite language, or a backend for whatever platform you desire.

## Integrated Debugger

FeatherGUI has a debugger built-in to the core library, which is accessible from any backend or program that chooses to expose it. You can use this debugger to inspect or modify your GUI layout on-the-fly, just like HTML, but without the headache. Edit, insert, delete, and re-order any element in your layout tree, or investigate why that one image isn't showing up. Like Chrome's HTML inspector, the debugger will highlight the margin, padding, and inner client area of any element, or you can press the Alt key and hover over an element in your GUI to find where it is in the layout tree.

## Layout Editor

Never again will you need to fiddle with code or write your own layout specification just to create you application's UI. FeatherGUI comes with a layout editor that can function with any compatible backend to create and edit UI layouts and custom skins for your game or application. Unlike Qt's layout editor, or many other GUI layout editors, FeatherGUI saves your layout seperate from behavior, so you can save your layout in an XML file and then load it back up using any language you want. You can even write your own serializer or parser for whatever file format you'd like to serialize the layout and skins to.

## Open-source and Free

The FeatherGUI library itself will always be free to use, even for commercial products. It's licensed under the permissive Apache License Version 2.0 to ensure anyone can use it without having to worry about legal trouble. **[Join the discussion on #feathergui](https://discord.gg/nFczp8J) in our official Discord Server!**

The library is in very early alpha stages of development right now, so please consider contributing if there's a feature you would like! FeatherGUI currently compiles on Windows and Linux, but only has a working backend for Windows (via Direct2D). More work is needed for an OpenGL-based backend, and native control backends for common operating systems.

Remember, your GUI toolkit should work for you, not the other way around.

## The Future

FeatherGUI has the potential to provide a true, universal GUI interface platform, allowing programs on any platform and any system to work together in ways that are impossible today. No longer will GUIs be limited to windows or programs, but instead will be built from encapsulated, data-bound windows, freely embeddable in every conceivable circumstance. Games and other specialty software will finally be able to interop directly with the native system UI and interact with other programs freely.

Join the revolution. Break free from the confines of widget toolkits and unleash your app's true potential.