import IAuthProvider from '../../../domain/providers/IAuthenticationProvider'
import IAccountProvider from '../../../domain/providers/IAccountProvider'
import { AuthType } from '../../../domain/models/auth/AuthType'
import AuthRequest from '../../../domain/models/auth/AuthRequest'
import { compareSync } from 'bcrypt'

export default class EmailAuthProvider implements IAuthProvider {
    private accountProvider: IAccountProvider

    constructor(accountProvider: IAccountProvider) {
        this.accountProvider = accountProvider
    }

    supports(type: AuthType): boolean {
        return type === AuthType.MAIL
    }

    async verifyLogin(request: AuthRequest): Promise<boolean> {
        const account = await this.accountProvider.findByEmail(request.email)
        return !(
            !account.password ||
            !compareSync(request.password, account.password)
        )
    }
}
