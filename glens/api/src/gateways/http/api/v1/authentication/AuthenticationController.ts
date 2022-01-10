import { FastifyInstance } from 'fastify'
import JWTToken from '../../../../../domain/models/auth/JWTToken'
import AuthenticationService from '../../../../../domain/services/AuthenticationService'
import Account from '../../../../../domain/models/account/Account'

export default class AuthenticationController {
    private service: AuthenticationService

    private router: FastifyInstance

    constructor(router: FastifyInstance, service: AuthenticationService) {
        this.router = router

        this.service = service

        router.post('/api/v1/auth/register/email', this.register.bind(this))

        router.post('/api/v1/auth/login/email', this.login.bind(this))

        router.post('/api/v1/auth/refresh', this.refreshToken.bind(this))
    }

    async login(
        req: { body: { email: string; password: string } },
        res: any
    ): Promise<JWTToken> {
        return this.service.authenticateByEmail(
            req.body.email,
            req.body.password
        )
    }

    async refreshToken(
        req: { body: { refreshToken: string } },
        res: any
    ): Promise<JWTToken> {
        return this.service.refreshToken(req.body.refreshToken)
    }

    async register(
        req: { body: { email: string; password: string } },
        res: any
    ): Promise<Account> {
        return this.service.registerByEmail(req.body.email, req.body.password)
    }
}
