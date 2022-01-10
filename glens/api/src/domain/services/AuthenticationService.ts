import IAccountProvider from '../providers/IAccountProvider'
import Account from '../models/account/Account'
import { IConfigurationProvider } from '../providers/IConfigurationProvider'
import JWTToken from '../models/auth/JWTToken'
import AuthRequest from '../models/auth/AuthRequest'
import { AuthType } from '../models/auth/AuthType'
import IAuthProvider from '../providers/IAuthenticationProvider'
import ITokenProvider from '../providers/ITokenProvider'

export default class AuthenticationService {
    private accountProvider: IAccountProvider

    private tokenProvider: ITokenProvider

    private authProviders: IAuthProvider[]

    constructor(
        accountProvider: IAccountProvider,
        tokenProvider: ITokenProvider,
        authProviders: IAuthProvider[]
    ) {
        this.accountProvider = accountProvider
        this.tokenProvider = tokenProvider
        this.authProviders = authProviders
    }

    async registerByEmail(email: string, password: string): Promise<Account> {
        return this.accountProvider.register(email, password)
    }

    async authenticateByEmail(
        email: string,
        password: string
    ): Promise<JWTToken> {
        const request = new AuthRequest(AuthType.MAIL, email, password)
        if (
            !(await this.authProviders
                .find(elt => elt.supports(request.type))
                ?.verifyLogin(request))
        ) {
            throw Error('InvalidPassword') // TODO : accurate error
        }
        const account = await this.accountProvider.findByEmail(email)
        return this.tokenProvider.generateTokens(
            account.id,
            account.secret,
            account.authority
        )
    }

    async refreshToken(refreshToken: string): Promise<JWTToken> {
        const auth = await this.tokenProvider.unpackToken(refreshToken)
        if (!auth.isRefreshToken) {
            throw Error('WrongToken') // TODO : accurate error
        }
        const account = await this.accountProvider.findById(auth.subject)
        await this.tokenProvider.verifyToken(refreshToken, account.secret)
        return this.tokenProvider.generateTokens(
            account.id,
            account.secret,
            account.authority
        )
    }

    async getConnectedAccount(authorization: string): Promise<Account> {
        const accessToken = authorization.replace('Bearer ', '')
        const auth = await this.tokenProvider.unpackToken(accessToken)
        if (auth.isRefreshToken) {
            throw Error('WrongToken') // TODO : accurate error
        }
        const account = await this.accountProvider.findById(auth.subject)
        await this.tokenProvider.verifyToken(accessToken, account.secret)
        return account
    }
}
