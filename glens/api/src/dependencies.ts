import { environment, Environment } from './environment'

/**
 * Modules
 */
import { HttpGateway } from './gateways/http/HttpGateway'
import { MongoDB } from './data/mongodb/MongoDB'
import { WebSocketGateway } from './gateways/ws/WebSocketGateway'
import { ImageController } from './gateways/ws/messages/v1/images/ImageController'
import { MessageBus } from './data/redis/MessageBus'
import { ImageProcessingService } from './domain/services/ImageProcessingService'
import AuthenticationController from './gateways/http/api/v1/authentication/AuthenticationController'
import AuthenticationService from './domain/services/AuthenticationService'
import { AccountProvider } from './data/mongodb/services/AccountProvider'
//import {TokenProvider} from '../build/src/data/security/TokenProvider'
import JwtTokenProvider from './data/security/JwtTokenProvider'
import EmailAuthProvider from './data/mongodb/services/EmailAuthProvider'
import { ExcelDatabase } from './data/Excel/Excel'
import PeopleController from './gateways/http/api/v1/people/PeopleController'
import { DataService } from './domain/services/DataService'
import { MongoDataProvider } from './data/mongodb/services/MongoDataProvider'
import {FaceDataController} from "./gateways/ws/messages/v1/facedata/FaceDataController";
import {FaceDataProcessingService} from "./domain/services/FaceDataProcessingService";

/**
 * Data
 */

/**
 * Domain
 */

export abstract class IModule {
    abstract start(): Promise<void>
    abstract stop(): Promise<void>
}

export class Dependencies {
    environment: Environment
    modules: IModule[] = []

    constructor(environment: Environment) {
        this.environment = environment
    }
}

export async function dependencies(
    denv?: Environment,
    deps: Dependencies = new Dependencies(denv || environment)
): Promise<Dependencies> {
    const env = await deps.environment.init()

    // Modules
    const database = new MongoDB(env)
    //const storage=new ExcelDatabase("./demo.xlsx")
    const storage = new MongoDataProvider()
    const http = new HttpGateway(env)
    const websockets = new WebSocketGateway(env)

    // Data
    const accountProvider = new AccountProvider()
    const tokenProvider = new JwtTokenProvider(env)
    const authProvider = new EmailAuthProvider(accountProvider)

    // Services & Providers
    const bus = new MessageBus(env)
    const service = new ImageProcessingService(bus)
    const faceDataService = new FaceDataProcessingService(bus, storage)
    const authService = new AuthenticationService(
        accountProvider,
        tokenProvider,
        [authProvider]
    )
    const dataService = new DataService(storage)
    // Controllers
    new ImageController(websockets, service, [storage])
    new FaceDataController(websockets, faceDataService)
    new AuthenticationController(http.router, authService)
    new PeopleController(http.router, dataService)
    // Initializable modules
    deps.modules.push(database)
    deps.modules.push(http)
    deps.modules.push(websockets)

    return deps
}

/*
{ id:, hair, age ... }
-> saveEvent({ person_id:, hair, age, date }) -> PersonEvent
-> processUser({ person_id:, hair, age, date }) -> mettre à jour le document Person


GET /people/events?from=date&to=date -> PersonEvent
[/people/events?from=date&to=date
    { person_id: 1, hair, age, date: "2021-02-05T09:42:16.499Z" },
    { person_id: 1, hair, age, date: "2021-02-05T09:45:16.499Z" }
]

GET /people -> Person
[
    { person_id: 1, age: [20, 25] }
]

**/
