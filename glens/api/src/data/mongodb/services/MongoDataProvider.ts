import { forClass } from '../../../domain/helpers/Logging'
import { DBData, DataDocument } from '../schemas/DataSchema'
import Data from '../../../domain/models/processing/Data'
import { IPeopleStorage } from '../../IPeopleStorage'

let logger = forClass('Mongo')
let mode_function =
    'function(a){return Object.values(a.reduce((count, e) => {if (!(e in count)) {count[e] = [0, e];}count[e][0]++;return count;}, {})).reduce((a, v) => v[0] < a[0] ? a : v, [0, null])[1];}'

export class MongoDataProvider implements IPeopleStorage {
    private static toData(doc: DataDocument): Data {
        return new Data(
            doc.person_id,
            doc.age,
            doc.gender,
            doc.date,
            doc.hairColor,
            doc.pose_data,
            doc.mask,
            doc.glasses
        )
    }

    private static fromData(data: Data): object {
        return {
            person_id: data.id,
            gender: data.gender,
            age: data.age,
            date: data.date,
            hairColor: data.hairColor,
            pose_data: data.pose_data,
            mask: data.mask,
            glasses: data.glasses
        }
    }

    fusePredictions(data: Object[]) {
        for (let id in data) {
            let gender_list = data[id]['genderList']
            let hairColor_list = data[id]['hairColorList']

            let gender_occurence = { count_max: 0, gender: '' }
            let hairColor_occurence = { count_max: 0, hairColor: '' }
            for (let i in gender_list) {
                gender_occurence[gender_list[i]] =
                    (gender_occurence[gender_list[i]] &&
                        gender_occurence[gender_list[i]] + 1) ||
                    1
                hairColor_occurence[hairColor_list[i]] =
                    (hairColor_occurence[gender_list[i]] &&
                        hairColor_occurence[gender_list[i]] + 1) ||
                    1

                if (
                    gender_occurence[gender_list[i]] >
                    gender_occurence.count_max
                ) {
                    gender_occurence.count_max =
                        gender_occurence[gender_list[i]]
                    gender_occurence.gender = gender_list[i]
                }

                if (
                    hairColor_occurence[hairColor_list[i]] >
                    hairColor_occurence.count_max
                ) {
                    hairColor_occurence.count_max =
                        hairColor_occurence[hairColor_list[i]]
                    hairColor_occurence.hairColor = hairColor_list[i]
                }
            }

            data[id]['gender'] = gender_occurence.gender
            data[id]['hairColor'] = hairColor_occurence.hairColor
        }
        return data
    }

    async getSummaryInfoOfPeopleBetweenDatesJS(from: Date, to: Date) {
        const data = await DBData.aggregate([
            {
                $match: { date: { $gte: from, $lte: to } },
            },
            {
                $group: {
                    _id: '$person_id',
                    person_id: {
                        $first: '$person_id',
                    },
                    age: {
                        $avg: '$age',
                    },
                    genderList: {
                        $push: '$gender',
                    },
                    hairColorList: {
                        $push: '$hairColor',
                    },
                },
            },
        ])

        if (!data) {
            throw Error('DataNotFound')
        }
        const fused_data = this.fusePredictions(data) as DataDocument[]

        let returnData = {}
        let event_count = 0

        for (let doc of fused_data) {
            returnData[event_count++] = MongoDataProvider.fromData(
                MongoDataProvider.toData(doc)
            )
        }

        return returnData
    }

    async getSummaryInfoOfPeopleBetweenDates(from: Date, to: Date) {
        const data = ((await DBData.aggregate([
            {
                //First Stage date filtering
                $match: { date: { $gte: from, $lte: to } },
            },
            {
                /**Group by person_id and calculate average age*/
                $group: {
                    _id: '$person_id',
                    person_id: {
                        $first: '$person_id',
                    },
                    age: {
                        $avg: '$age',
                    },
                    genderList: {
                        $push: '$gender',
                    },
                    hairColorList: {
                        $push: '$hairColor',
                    },
                },
            },
            {
                $addFields: {
                    gender: {
                        $function: {
                            body: mode_function,
                            args: ['$genderList'],
                            lang: 'js',
                        },
                    },

                    hairColor: {
                        $function: {
                            body: mode_function,
                            args: ['$hairColorList'],
                            lang: 'js',
                        },
                    },
                },
            },
            {
                $unset: ['hairColorlist', 'genderlist'],
            },
        ])) as any) as DataDocument[]

        if (!data) {
            throw Error('DataNotFound')
        }

        let returnData = {}
        let event_count = 0

        for (let doc of data) {
            returnData[event_count++] = MongoDataProvider.fromData(
                MongoDataProvider.toData(doc)
            )
        }

        return returnData
    }

    async getPositionInfoOfAPerson(from: Date, to: Date, person_id: string) {
        const data = await DBData.aggregate([
            {
                $match: {
                    date: { $gte: from, $lte: to },
                    person_id: { $eq: person_id },
                },
            },
            {
                $group: {
                    _id: '$person_id',
                    person_id: {
                        $first: '$person_id',
                    },
                    age: {
                        $avg: '$age',
                    },
                    gender: {
                        $first: '$gender',
                    },
                    hairColor: {
                        $first: '$hairColor',
                    },
                    poseList: {
                        $push: '$pose_data',
                    },
                    dateList: {
                        $push: '$date',
                    },
                },
            },
        ])

        if (!data) {
            throw Error('DataNotFound')
        }
        logger.info(data)

        let returnData = {}
        let event_count = 0

        for (let doc of data) {
            returnData[event_count++] = MongoDataProvider.fromData(
                MongoDataProvider.toData(doc)
            )
            returnData[event_count - 1]['pose_data'] = doc['poseList']
            returnData[event_count - 1]['timestamps'] = doc['dateList']
        }

        return returnData
    }

    async getPeopleDataBetweenDates(from: Date, to: Date) {
        const data = (await DBData.find({
            date: { $gte: from, $lte: to },
        })) as DataDocument[]
        if (!data) {
            throw Error('DataNotFound')
        }

        let returnData = {}
        let event_count = 0
        for (let doc of data) {
            returnData[event_count++] = MongoDataProvider.fromData(
                MongoDataProvider.toData(doc)
            )
            delete returnData[event_count - 1]['pose_data']
        }

        return returnData
    }

    async savePeopleData(message: string) {
        let parsedMessage = JSON.parse(message).data

        for (let personId in parsedMessage) {
            if (isNaN(parseFloat(personId))) {
                //not recognized/ no recognition/ do not store the values
                continue
            }

            await DBData.create(
                MongoDataProvider.fromData(
                    new Data(
                        personId,
                        parsedMessage[personId].age,
                        parsedMessage[personId].sex,
                        new Date(),
                        parsedMessage[personId].hairColor,
                        parsedMessage[personId].person_pose_data,
                        undefined, // TODO
                        undefined
                    )
                )
            )
        }
    }

    async saveData(data: Data[]) {
        await DBData.create(
            data.map(elt => MongoDataProvider.fromData(elt))
        )
    }
}
