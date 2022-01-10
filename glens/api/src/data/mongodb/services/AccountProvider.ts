import IAccountProvider from '../../../domain/providers/IAccountProvider'
import Account from '../../../domain/models/account/Account'
import { AccountDocument, DBAccount } from '../schemas/AccountSchema'
import { hashSync } from 'bcrypt'
import * as uuid from 'uuid'
import { DomainAccessRole } from '../../../domain/models/processing/nested/DomainAccessRole'

export class AccountProvider implements IAccountProvider {
    private static toAccount(doc: AccountDocument): Account {
        return new Account(
            doc.id,
            doc.email,
            doc.password,
            doc.createdDate,
            doc.secret,
            doc.authority
        )
    }

    private static fromAccount(account: Account): object {
        return {
            id: account.id,
            email: account.email,
            password: account.password,
            createdDate: account.createdDate,
            secret: account.secret,
            authority: account.authority,
        }
    }

    async findByEmail(email: string): Promise<Account> {
        const account = (await DBAccount.findOne({
            email: email,
        })) as AccountDocument
        if (!account) {
            throw Error('AccountNotFound')
        }
        return AccountProvider.toAccount(account)
    }

    async findById(id: string): Promise<Account> {
        const account = (await DBAccount.findById(id)) as AccountDocument
        if (!account) {
            throw Error('AccountNotFound')
        }
        return AccountProvider.toAccount(account)
    }

    async register(email: string, password: string): Promise<Account> {
        // Check if account already exists
        const acc = (await DBAccount.find({
            email: email,
        })) as AccountDocument[]
        if (acc.length > 0) {
            throw Error('AccountAlreadyCreated')
        }
        return AccountProvider.toAccount(
            (await DBAccount.create(
                AccountProvider.fromAccount(
                    new Account(
                        '',
                        email,
                        hashSync(password, 10),
                        new Date(),
                        uuid.v1(),
                        DomainAccessRole.USER
                    )
                )
            )) as AccountDocument
        )
    }

    async update(account: Account): Promise<Account> {
        const accountEmail = await DBAccount.findOne({ email: account.email })
        if (accountEmail && accountEmail.id != account.id) {
            throw Error('EmailAccountAlreadyUsed')
        }
        return AccountProvider.toAccount(
            (await DBAccount.findByIdAndUpdate(
                account.id,
                AccountProvider.fromAccount(account)
            )) as AccountDocument
        )
    }
}
