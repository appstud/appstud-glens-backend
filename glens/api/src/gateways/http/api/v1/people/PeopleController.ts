import { FastifyInstance } from 'fastify'
import { DataService } from '../../../../../domain/services/DataService'
import { forClass } from '../../../../../domain/helpers/Logging'

export default class PeopleController {
    private service: DataService
    private router: FastifyInstance
    private logger = forClass('PeopleController')

    constructor(router: FastifyInstance, service: DataService) {
        this.router = router
        this.service = service

        router.get('/api/v1/people/events', this.getDataBetweenDates.bind(this))
        router.get(
            '/api/v1/people/events/groupById',
            this.getSummaryInfoOfPeopleBetweenDates.bind(this)
        )
        router.get(
            '/api/v1/positionOfPerson',
            this.getPositionInfoOfAPerson.bind(this)
        )
    }

    kill(error: string) {
        throw new Error(error)
    }

    async getPositionInfoOfAPerson(request, reply) {
        let from =
            request.query.from ||
            this.kill("Please provide a 'from' query parameter")
        let to =
            request.query.to ||
            this.kill("Please provide a 'to' query parameter")
        let person_id =
            request.query.person_id ||
            this.kill("Please provide a 'person_id' query parameter")

        return this.service.getPositionInfoOfAPerson(
            new Date(from),
            new Date(to),
            person_id as string
        )
    }

    async getSummaryInfoOfPeopleBetweenDates(request, reply) {
        let from =
            request.query.from ||
            this.kill("Please provide a 'from' query parameter")
        let to =
            request.query.to ||
            this.kill("Please provide a 'to' query parameter")
        return this.service.getSummaryInfoOfPeopleBetweenDates(
            new Date(from),
            new Date(to)
        )
    }
    async getDataBetweenDates(request, reply) {
        let from =
            request.query.from ||
            this.kill("Please provide a 'from' query parameter")
        let to =
            request.query.to ||
            this.kill("Please provide a 'to' query parameter")
        return this.service.getDataBetweenDates(new Date(from), new Date(to))
    }
}
