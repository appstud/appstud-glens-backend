import { dependencies } from './dependencies'
;(async function main() {
    const deps = await dependencies()
    for (const it of deps.modules) await it.start()
})()
