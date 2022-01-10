import { dependencies, Dependencies } from '../src/dependencies'
import environment from '../src/environment'

let deps: Dependencies
let http

beforeAll(async () => {
    const env = await environment.init('test')
    deps = await dependencies(env)
    for (const it of deps.modules) await it.start()
})

afterAll(async () => {
    for (const it of deps.modules) await it.stop()
    //await deps.gateways.database.remove()
})
