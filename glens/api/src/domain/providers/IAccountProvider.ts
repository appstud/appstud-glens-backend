import Account from '../models/account/Account'

export default interface IAccountProvider {
    register(email: String, password: String): Promise<Account>

    findById(id: string): Promise<Account>

    findByEmail(email: string): Promise<Account>

    update(account: Account): Promise<Account>
}
