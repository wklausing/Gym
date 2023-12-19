class EdgeServer:

    def __init__(self, id, location, price=1, bandwidth_kbps=1000):
        self.id = id
        self.location = location
        self.price = price
        self.contigent = 0
        self.bandwidth_kbps = bandwidth_kbps
        self.clients = []
        self.money = 0

    def sellContigent(self, cpMoney, amount):
        price = self.price * amount
        if cpMoney >= price:
            cpMoney -= price
            self.money += price
            self.contigent += amount * 1000
        return round(cpMoney, 2)
    
    def addClient(self, client):
        self.clients.append(client)
        self.currentBandwidth = self.bandwidth_kbps / len(self.clients)

    def removeClient(self, client):
        self.clients.remove(client)
        if len(self.clients) == 0: return
        self.currentBandwidth = self.bandwidth_kbps / len(self.clients)

    def deductContigent(self, amount):
        amountMB = amount / 8_000_000
        if self.contigent >= amountMB:
            self.contigent -= amountMB
            self.contigent = round(self.contigent, 2)
        else:
            gym.logger.warn('Deducting too much contigent from CDN %s.' % self.id)
            quit()
    
    @property
    def bandwidth(self):
        return self.bandwidth_kbps