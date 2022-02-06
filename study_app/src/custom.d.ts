
// see: https://webpack.js.org/guides/typescript/#importing-other-assets

declare module '*.jpg' {
    const content: any,
    export default content;
}

declare module "*.svelte" {
    const value: any;
    export default value;
}