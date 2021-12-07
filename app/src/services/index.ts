import { DataStore } from '@marcellejs/core';
import { MyService, Message, getIdService } from './MyService';

export function myService<T>(store: DataStore, name: string) {
    return store.service(name) as unknown as MyService<T>
}

export { MyService, getIdService };
export type { Message };