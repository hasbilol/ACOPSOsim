import turtle
import random

def draw_petal(t, x, y, petal_color):
    t.penup()
    t.goto(x, y)
    t.pendown()
    t.color(petal_color)
    t.begin_fill()
    for _ in range(2):
        t.circle(30, 60)
        t.left(120)
        t.circle(30, 60)
        t.left(120)
    t.end_fill()

def draw_flower(x, y):
    t = turtle.Turtle()
    t.speed(10)
    t.hideturtle()
    
    # Draw petals
    for angle in range(0, 360, 45):
        t.setheading(angle)
        draw_petal(t, x, y, "white")
    
    # Draw center
    t.penup()
    t.goto(x, y - 10)
    t.pendown()
    t.color("yellow")
    t.begin_fill()
    t.circle(10)
    t.end_fill()

def draw_stem(x, y):
    t = turtle.Turtle()
    t.speed(10)
    t.hideturtle()
    t.color("green")
    t.pensize(5)
    
    t.penup()
    t.goto(x, y - 10)
    t.pendown()
    t.goto(x, y - 100)

def draw_bouquet():
    screen = turtle.Screen()
    screen.bgcolor("#262626")
    
    # Randomly place flowers
    positions = [(-100, 50), (-50, 70), (0, 90), (50, 70), (100, 50)]
    for x, y in positions:
        draw_flower(x, y)
        draw_stem(x, y - 10)
    
    screen.exitonclick()
    turtle.done()

draw_bouquet()
