(function (root) {
    'use strict';
    function createQueue(limit = 12) {
        const items = [];
        let nextId = 0;
        let running = false;
        function idle() {
            if (running) throw new Error('The queue is running.');
        }
        function find(id) {
            const item = items.find(entry => entry.id === id);
            if (!item) throw new Error('This crop is no longer in the queue.');
            return item;
        }
        function addMany(crops) {
            idle();
            if (crops.length > limit - items.length) throw new Error(`The queue holds up to ${limit} crops.`);
            const added = crops.map((crop, index) => ({ ...crop, id: nextId + index + 1, status: 'pending' }));
            nextId += added.length;
            items.push(...added);
            return added.map(item => item.id);
        }
        return {
            get items() { return items.map(item => ({ ...item })); },
            get running() { return running; },
            get full() { return items.length >= limit; },
            get remaining() { return limit - items.length; },
            add(crop) {
                return addMany([crop])[0];
            },
            addMany,
            approve(id, request) {
                idle();
                Object.assign(find(id), { status: 'approved', request, result: undefined });
            },
            skip(id) {
                idle();
                Object.assign(find(id), { status: 'skipped', request: undefined, result: undefined });
            },
            review(id) {
                idle();
                Object.assign(find(id), { status: 'pending', request: undefined, result: undefined });
            },
            remove(id) {
                idle();
                items.splice(items.indexOf(find(id)), 1);
            },
            async run(restore) {
                idle();
                if (items.some(item => item.status === 'pending' || item.status === 'failed')) {
                    throw new Error('Review or skip every crop before running the queue.');
                }
                running = true;
                try {
                    for (const item of items) {
                        if (item.status !== 'approved') continue;
                        item.status = 'running';
                        try {
                            item.result = await restore(item.request, { ...item });
                            item.status = 'completed';
                        } catch (error) {
                            item.status = 'failed';
                            throw error;
                        }
                    }
                } finally {
                    running = false;
                }
            }
        };
    }
    if (typeof module !== 'undefined' && module.exports) module.exports = { createQueue };
    else root.createRestorationQueue = createQueue;
})(globalThis);
