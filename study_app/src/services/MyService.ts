
import { ServiceMethods, Params, Id, NullableId } from '@feathersjs/feathers';

export interface Message<T> {
    content: {
        user: {
            id: Id;
        };
        task: {
            name: string;
            condition: string;
            iteration: number;
        };
        data: T;
    };
    // those are added by feathers
    id?: Id; 
    createdAt?: Date;
    updatedAt?: Date;
}

export function getIdService(participantId: Id, condition: string) {
  return `${participantId}-${condition}`;
}

export class MyService<T> implements ServiceMethods<Message<T>> {
    static async save<T>(serv: MyService<T>, participantId: Id, condition: string, iteration: number, taskName: string, data: T): Promise<Message<T>> {
        return serv.create({
            content: {
                data: data,
                user: {
                    id: participantId
                },
                task: {
                    condition: condition,
                    name: taskName,
                    iteration: iteration
                }
            }
        }, {});
    }

    async find(params: Params): Promise<any> {
        // todo
        return Object.values(this.messages).map(msg => {
            data: msg
        });
    }

    async create(data: Message<T>, params: Params): Promise<Message<T>> {
        const l = this.messages.push(data);
        return this.messages[l-1];
    }

    // todo?
    async get(id: Id, params: Params): Promise<Message<T>> { return null; }
    async update(id: NullableId, data: Message<T>, params: Params): Promise<Message<T>> { return null; }
    async patch(id: NullableId, data: Message<T>, params: Params): Promise<Message<T>> { return null; }
    async remove(id: NullableId, params: Params): Promise<Message<T>> { return null; }

    private messages: Array<Message<T>>;
}