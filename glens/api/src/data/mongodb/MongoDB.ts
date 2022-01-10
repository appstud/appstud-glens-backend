import * as mongoose from 'mongoose'
import { Environment } from '../../environment'
import { IModule } from '../../dependencies'
import { forClass } from '../../domain/helpers/Logging'
import { sleep } from '../../domain/helpers/Functions'
const Migrator = require('migrate-mongoose')

const MIGRATIONS_FOLDER = `${__dirname}/migrations/`

export class MongoDB implements IModule {
    private logger = forClass('MongoDB')
    private environnement: Environment
    private uri: string
    private retries: number
    private retryDelay: number

    constructor(environnement: Environment) {
        this.environnement = environnement
        this.uri = this.environnement.getConfigurationOrDefault<string>(
            'MONGODB_URI',
            'mongodb://127.0.0.1:27017/glens-backend'
        )
        this.retries = +this.environnement.getConfigurationOrDefault<number>(
            'MONGODB_RETRIES',
            3
        )
        this.retryDelay = +this.environnement.getConfigurationOrDefault<number>(
            'MONGODB_RETRY_DELAY',
            3000
        )
    }

    private async migrate(con: mongoose.Mongoose) {
        const migrator = new Migrator({
            connection: con.connection,
            migrationsPath: MIGRATIONS_FOLDER,
            autosync: true,
        })
        const migrations = await migrator.list()
        if (!migrations.find((m: { state: string }) => m.state === 'down'))
            return this.logger.debug('No runnable migrations found')
        await migrator.run('up')
    }

    private async connect(tries = 0): Promise<void> {
        try {
            const connection = await mongoose.connect(this.uri, {
                useNewUrlParser: true,
                useUnifiedTopology: true,
            })
            await this.migrate(connection)
        } catch (e) {
            this.logger.error(`Unable to connect to mongo ${this.uri}:`, e)
            if (tries < this.retries) {
                this.logger.error(
                    `Waiting ${
                        this.retryDelay
                    } ms before retrying ... (Retry ${tries + 1} / ${
                        this.retries
                    })`
                )
                await sleep(this.retryDelay)
                return this.connect(tries + 1)
            }
            throw e
        }
    }

    async start() {
        await this.connect()
        return
    }

    async stop() {
        await mongoose.disconnect()
    }

    async remove() {
        if (!this.environnement.isTest)
            throw new Error(
                'You cannot remove programmaticaly database is environement is not test. Aborted.'
            )

        // If we are connected - directly drop database
        if (mongoose.connection.readyState === 1)
            return mongoose.connection.dropDatabase()

        // Create connection and drop database if not connected
        const connection = await mongoose.connect(this.uri)
        await connection.connection.dropDatabase()
        await connection.disconnect()
    }
}
